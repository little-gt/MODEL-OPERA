# -*- coding: utf-8 -*-

import os

"""
OPERA/opera_analysis/visualizer.py

This module provides the AnalysisVisualizer class, which is used to create
scientific plots from the model's predictions to help with interpretation.
"""
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from typing import Dict, Any, Callable

from opera_data.composition import CompositionEngine


class AnalysisVisualizer:
    """
    A class to create visualizations for the model's analysis,
    including reconstructions, residuals, and component decompositions.
    """

    def __init__(self, phenomenon_map: Dict[str, Callable]):
        """
        Initializes the visualizer.

        Args:
            phenomenon_map (Dict[str, Callable]): A dictionary mapping the
                string names of phenomena to the actual callable functions.
        """
        self.phenomenon_map = phenomenon_map
        self.composition_engine = CompositionEngine()

    def _reconstruct_image(self, predicted_label: Dict[str, Any]) -> np.ndarray:
        """
        Internal helper to reconstruct an image from a predicted label.
        """
        components_recipe = predicted_label.get('components', [])

        # Translate the recipe from string names to callable functions
        engine_components = []
        for comp in components_recipe:
            func_name = comp.get('phenomenon_name')
            if func_name in self.phenomenon_map:
                # Ensure params dict exists even if empty
                params = comp.get('params', {})
                engine_components.append({
                    'phenomenon': self.phenomenon_map[func_name],
                    'params': params
                })
            else:
                print(f"Warning: Phenomenon function '{func_name}' not found in map. Skipping.")

        if not engine_components:
            res = predicted_label.get('resolution_px', 256)
            return np.zeros((res, res))

        reconstructed_img = self.composition_engine.compose(
            engine_components,
            composition_type=predicted_label.get('composition_type', 'sequential')
        )
        return reconstructed_img

    def plot_reconstruction_and_residual(
            self,
            original_image: np.ndarray,
            predicted_label: Dict[str, Any],
            save_path: str = None
    ):
        """
        Plots the original image, the model's reconstruction, and the residual difference.

        Args:
            original_image (np.ndarray): The initial image that was analyzed.
            predicted_label (Dict[str, Any]): The model's JSON output as a dictionary.
            save_path (str, optional): If provided, saves the plot to this path.
        """
        reconstructed_image = self._reconstruct_image(predicted_label)

        # Normalize original image if it's not in [0, 1] range
        if original_image.max() > 1.0:
            original_image = original_image / 255.0

        residual = np.abs(original_image - reconstructed_image)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Model Analysis: Reconstruction & Residual', fontsize=16)

        axes[0].imshow(original_image, cmap='gray')
        axes[0].set_title('Original Input Image')
        axes[0].axis('off')

        axes[1].imshow(reconstructed_image, cmap='gray')
        axes[1].set_title('Model Reconstruction')
        axes[1].axis('off')

        im = axes[2].imshow(residual, cmap='inferno')
        axes[2].set_title('Residual (Difference)')
        axes[2].axis('off')
        fig.colorbar(im, ax=axes[2], shrink=0.8)

        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_decomposition(self, predicted_label: Dict[str, Any], save_path: str = None):
        """
        Decomposes the prediction into its constituent parts and plots them individually.

        Args:
            predicted_label (Dict[str, Any]): The model's JSON output as a dictionary.
            save_path (str, optional): If provided, saves the plot to this path.
        """
        components_recipe = predicted_label.get('components', [])
        num_components = len(components_recipe)
        if num_components == 0:
            print("No components in prediction to decompose.")
            return

        fig, axes = plt.subplots(1, num_components + 1, figsize=(6 * (num_components + 1), 5), constrained_layout=True)
        fig.suptitle('Model Prediction: Component Decomposition', fontsize=16)

        # Create a delta image (point source) for visualizing modifiers like aberrations
        res = predicted_label.get('resolution_px', 256)
        delta_image = np.zeros((res, res))
        delta_image[res // 2, res // 2] = 1.0

        for i, comp in enumerate(components_recipe):
            func_name = comp.get('phenomenon_name')
            func = self.phenomenon_map.get(func_name)
            params = comp.get('params', {})

            if func:
                # If the function is a modifier, apply it to a point source to see its PSF
                if func_name.startswith('apply_'):
                    component_img = func(delta_image, **params)
                    title = f"Component {i + 1}: {func_name}\n(Point Spread Function)"
                else:  # Otherwise, it's a generator
                    component_img = func(**params)
                    title = f"Component {i + 1}: {func_name}"

                ax = axes[i]
                ax.imshow(component_img, cmap='gray')
                ax.set_title(title, fontsize=10)
                ax.axis('off')

        # In the last panel, show the final combined result
        reconstructed_image = self._reconstruct_image(predicted_label)
        ax = axes[-1]
        ax.imshow(reconstructed_image, cmap='gray')
        ax.set_title('Final Combined Result')
        ax.axis('off')

        if save_path:
            plt.savefig(save_path)
        plt.show()


# When run directly, perform a test
if __name__ == '__main__':
    # Add project root to path to allow imports
    import sys

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(project_root)

    from opera_data.phenomena.diffraction import generate_airy_disk
    from opera_data.phenomena.aberrations import apply_zernike_aberration

    print("--- Testing Analysis Visualizer ---")

    # 1. Create the mapping from function names to functions
    phenomenon_map = {
        'generate_airy_disk': generate_airy_disk,
        'apply_zernike_aberration': apply_zernike_aberration
    }

    # 2. Instantiate the visualizer
    visualizer = AnalysisVisualizer(phenomenon_map)

    # 3. Create a fake "original image" and a fake "prediction"
    original_image_params = {'aperture_diameter_mm': 1.0, 'resolution_px': 256}
    original_image_aberrated = apply_zernike_aberration(
        generate_airy_disk(**original_image_params),
        {'zernike_coeffs': {4: 0.5}}  # with some defocus
    )

    predicted_label = {
        "image_id": "pred_001",
        "resolution_px": 256,
        "composition_type": "sequential",
        "components": [
            {
                "phenomenon_name": "generate_airy_disk",
                "params": {"aperture_diameter_mm": 1.0, "resolution_px": 256}
            },
            {
                "phenomenon_name": "apply_zernike_aberration",
                "params": {"zernike_coeffs": {4: 0.5}}  # Model correctly predicts defocus
            }
        ]
    }

    print("\n[1] Plotting Reconstruction and Residual...")
    visualizer.plot_reconstruction_and_residual(original_image_aberrated, predicted_label)

    print("\n[2] Plotting Component Decomposition...")
    visualizer.plot_decomposition(predicted_label)

    print("\n--- Visualizer Test Complete ---")