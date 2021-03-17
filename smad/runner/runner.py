import os
import subprocess
from statistics import mean
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import ImageGrid
from numpy import ndarray as NDArray
from skimage import measure
from sklearn.decomposition._dict_learning import sparse_encode
from sklearn.metrics import auc, roc_auc_score
from tqdm import tqdm

from smad.runner import BaseRunner


class Runner(BaseRunner):
    def _train(self) -> Tuple[NDArray, NDArray]:

        _, imgs, _ = next(iter(self.dataloaders["train"]))
        Y = self._convert_imgs_to_Y(imgs[:, :, :, 0].detach().numpy())
        num_patches, num_features = Y.shape
        D = np.random.rand(self.cfg.params.num_basis, num_features)
        D = np.dot(D, np.diag(1.0 / np.sqrt(np.diag(np.dot(D.T, D)))))

        errors = [np.inf]

        for epoch in tqdm(range(1, self.cfg.params.epochs + 1), desc="train"):
            X = sparse_encode(Y, D, algorithm="omp")
            for j in range(self.cfg.params.num_basis):
                nonzero_index = X[:, j] != 0
                X[nonzero_index, j] = 0
                error = Y[nonzero_index, :] - np.dot(X[nonzero_index, :], D)
                U, S, V = np.linalg.svd(error)
                X[nonzero_index, j] = U[:, 0] * S[0]
                D[j, :] = V.T[:, 0]

            errors.append(np.linalg.norm(Y - np.dot(X, D), "fro"))
            mlflow.log_metric("error", errors[-1], step=epoch)

            if np.abs(errors[-1] - errors[-2]) < self.cfg.params.tolerance:
                break

        return (D, X)

    def _test(self, D: NDArray, X: NDArray) -> None:

        artifacts = {"stem": [], "img": [], "mask": [], "reconstructed_img": [], "amap": []}
        for stem, img, mask in tqdm(self.dataloaders["test"], desc="test"):

            assert len(img) == 1, "dataloaders.test.args.batch_size must be 1"

            img = img[:, :, :, 0].detach().numpy()
            Y = self._convert_imgs_to_Y(img)
            num_patches, num_features = Y.shape
            X = sparse_encode(Y, D, algorithm="omp")

            reconstructed_Y = np.dot(X, D)
            reconstructed_img = self._convert_Y_to_img(reconstructed_Y)

            amap = np.abs(img.squeeze() - reconstructed_img)
            amap = self._mean_smooth(amap)

            artifacts["stem"].extend(stem)
            artifacts["img"].extend(self._denormalize(img))
            artifacts["mask"].extend(mask.detach().numpy())
            artifacts["reconstructed_img"].append(self._denormalize(reconstructed_img))
            artifacts["amap"].append(amap)

        amaps = np.array(artifacts["amap"])
        amaps = (amaps - amaps.min()) / (amaps.max() - amaps.min())

        roc_score = self._compute_roc_score(amaps, np.array(artifacts["mask"]))
        pro_score = self._compute_pro_score(amaps, np.array(artifacts["mask"]))

        mlflow.log_metrics({"roc_score": roc_score, "pro_score": pro_score})

        self._savegif(
            artifacts["stem"],
            np.array(artifacts["img"]),
            np.array(artifacts["mask"]),
            np.array(artifacts["reconstructed_img"]),
            amaps,
        )

    def _convert_imgs_to_Y(self, imgs: NDArray) -> NDArray:

        b, h, w = imgs.shape
        patch_size = self.cfg.params.patch_size
        Y = imgs.reshape(b, int(h / patch_size), patch_size, int(w / patch_size), patch_size)
        Y = Y.transpose(0, 1, 3, 2, 4)
        Y = Y.reshape(-1, patch_size * patch_size)
        return Y

    def _convert_Y_to_img(self, Y: NDArray) -> NDArray:

        h = self.cfg.params.height
        w = self.cfg.params.width
        patch_size = self.cfg.params.patch_size
        img = Y.reshape(int(h / patch_size), int(w / patch_size), patch_size, patch_size)
        img = img.transpose(0, 2, 1, 3)
        img = img.reshape(h, w)
        return img

    def _denormalize(self, img: NDArray) -> NDArray:

        img = (img * self.cfg.params.std + self.cfg.params.mean) * 255.0
        return img.astype(np.uint8)

    def _mean_smooth(self, amap: NDArray) -> NDArray:

        kernel = np.ones((5, 5)) / 25
        amap = cv2.filter2D(amap, -1, kernel)
        return amap

    def _compute_roc_score(self, amaps: NDArray, masks: NDArray) -> float:

        num_data = len(amaps)
        masks[masks != 0] = 1
        y_scores = amaps.reshape(num_data, -1).max(axis=1)
        y_trues = masks.reshape(num_data, -1).max(axis=1)
        return roc_auc_score(y_trues, y_scores)

    def _compute_pro_score(self, amaps: NDArray, masks: NDArray) -> float:

        df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
        binary_amaps = np.zeros_like(amaps, dtype=np.bool)

        max_step = 200
        min_th = amaps.min()
        max_th = amaps.max()
        delta = (max_th - min_th) / max_step

        for th in tqdm(np.arange(min_th, max_th, delta), desc="compute pro"):
            binary_amaps[amaps <= th] = 0
            binary_amaps[amaps > th] = 1

            pros = []
            for binary_amap, mask in zip(binary_amaps, masks):
                for region in measure.regionprops(measure.label(mask)):
                    axes0_ids = region.coords[:, 0]
                    axes1_ids = region.coords[:, 1]
                    TP_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                    pros.append(TP_pixels / region.area)

            inverse_masks = 1 - masks
            FP_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
            fpr = FP_pixels / inverse_masks.sum()

            df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

        return auc(df["fpr"], df["pro"])

    def _savegif(
        self,
        stems: List[str],
        imgs: NDArray,
        masks: NDArray,
        reconstructed_imgs: NDArray,
        amaps: NDArray,
    ) -> None:

        os.mkdir("results")
        pbar = tqdm(enumerate(zip(stems, imgs, masks, reconstructed_imgs, amaps)), desc="savegif")
        for i, (stem, img, mask, reconstructed_img, amap) in pbar:

            # How to get two subplots to share the same y-axis with a single colorbar
            # https://stackoverflow.com/a/38940369
            grid = ImageGrid(
                fig=plt.figure(figsize=(16, 4)),
                rect=111,
                nrows_ncols=(1, 4),
                axes_pad=0.15,
                share_all=True,
                cbar_location="right",
                cbar_mode="single",
                cbar_size="5%",
                cbar_pad=0.15,
            )

            grid[0].imshow(img, cmap="gray")
            grid[0].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
            grid[0].set_title("Input Image", fontsize=20)

            grid[1].imshow(reconstructed_img, cmap="gray")
            grid[1].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
            grid[1].set_title("Reconstructed Image", fontsize=20)

            grid[2].imshow(img, cmap="gray")
            grid[2].imshow(mask, alpha=0.3, cmap="Reds")
            grid[2].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
            grid[2].set_title("Ground Truth", fontsize=20)

            grid[3].imshow(img, cmap="gray")
            im = grid[3].imshow(amap, alpha=0.3, cmap="jet", vmin=0, vmax=1)
            grid[3].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
            grid[3].cax.toggle_label(True)
            grid[3].set_title("Anomaly Map", fontsize=20)

            plt.colorbar(im, cax=grid.cbar_axes[0])
            plt.savefig(f"results/{stem}.png", bbox_inches="tight")
            plt.close()

        # NOTE(inoue): The gif files converted by PIL or imageio were low-quality.
        #              So, I used the conversion command (ImageMagick) instead.
        subprocess.run("convert -delay 100 -loop 0 results/*.png result.gif", shell=True)
