"""
Color to Thickness Mapping Tool for Gold Flakes

This module provides a GUI application to create lookup tables that map RGB/LAB
color values of gold flakes observed under an optical microscope to their
corresponding thicknesses. It also allows mapping colors from images to
thickness using pre-created lookup tables.

Author: Daniel Abraham Elmaleh
Institution: EPFL - NAM Lab
Project: Anti-Stokes Spectroscopy of Crystalline Plasmonic Metals
Year: 2024
"""

import os
import threading
from typing import Optional, Tuple, Dict

import cv2
import numpy as np
import pandas as pd
from tkinter import (
    Tk, Button, Label, Entry,
    filedialog, messagebox, StringVar, Radiobutton
)


# =============================================================================
# Color Conversion Utilities
# =============================================================================

def rgb_to_lab(rgb: Tuple[int, int, int]) -> Optional[np.ndarray]:
    """
    Convert RGB color values to LAB color space.

    Args:
        rgb: Tuple of (R, G, B) values in range 0-255.

    Returns:
        LAB values as numpy array, or None if conversion fails.
    """
    try:
        lab = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB)
        return lab[0][0]
    except Exception as e:
        messagebox.showerror("Conversion Error", f"Failed to convert RGB to LAB: {e}")
        return None


def normalize_colors(
    sample_rgb: Tuple[float, ...],
    sample_lab: Tuple[float, ...],
    background_rgb: Tuple[float, ...],
    background_lab: Tuple[float, ...]
) -> Tuple[Optional[Tuple[float, ...]], Optional[Tuple[float, ...]]]:
    """
    Normalize sample colors relative to background reference.

    Args:
        sample_rgb: RGB values of the sample.
        sample_lab: LAB values of the sample.
        background_rgb: RGB values of the background reference.
        background_lab: LAB values of the background reference.

    Returns:
        Tuple of (normalized_rgb, normalized_lab), or (None, None) on error.
    """
    try:
        normalized_rgb = tuple(
            np.round(np.clip(np.array(sample_rgb) / np.array(background_rgb), 0, 255), 5)
        )
        normalized_lab = tuple(
            np.round(np.clip(np.array(sample_lab) / np.array(background_lab), 0, 255), 5)
        )
        return normalized_rgb, normalized_lab
    except Exception as e:
        messagebox.showerror("Normalization Error", f"Failed to normalize colors: {e}")
        return None, None


# =============================================================================
# ROI Selection
# =============================================================================

def select_roi_adjustable(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Display image and allow user to select a Region of Interest (ROI).

    Args:
        image: OpenCV image array (BGR format).

    Returns:
        ROI coordinates as (x, y, width, height), or None if cancelled.
    """
    MAX_DISPLAY_SIZE = 800

    height, width = image.shape[:2]
    if max(width, height) > MAX_DISPLAY_SIZE:
        scale = MAX_DISPLAY_SIZE / max(width, height)
        image = cv2.resize(image, (int(width * scale), int(height * scale)))

    cv2.namedWindow('Select ROI', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Select ROI', 800, 600)
    roi = cv2.selectROI('Select ROI', image, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    return roi if roi != (0, 0, 0, 0) else None


# =============================================================================
# Main Application Class
# =============================================================================

class ColorToThicknessApp:
    """
    GUI Application for color-to-thickness mapping of gold flakes.

    This application provides two main functionalities:
    1. Creating lookup tables by correlating colors with known thicknesses
    2. Mapping unknown sample colors to thickness using existing lookup tables
    """

    # Supported substrate types
    SUBSTRATE_TYPES = ['Float', 'Borofloat', 'Si', 'D263']

    def __init__(self, root: Tk):
        """
        Initialize the application.

        Args:
            root: Tkinter root window.
        """
        self.root = root
        self.root.title("Lookup Table Creation / Color to Thickness Mapping")
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        # State variables
        self.background_image_loaded = False
        self.goldflake_image_loaded = False
        self.background_rgb: Optional[Tuple[int, int, int]] = None
        self.background_lab: Optional[Tuple[int, int, int]] = None
        self.gold_flake_rgb: Optional[Tuple[int, int, int]] = None
        self.gold_flake_lab: Optional[Tuple[int, int, int]] = None
        self.selected_image_name = ""

        # Build UI
        self._create_widgets()
        self._update_entry_options()

    def _create_widgets(self):
        """Create and layout all GUI widgets."""
        # Radio button groups
        self.function_choice = self._create_radiobutton_group(
            "Functionality:",
            ["Lookup Table Entry", "Map Image to Lookup Table"],
            row=0
        )
        self.entry_choice = self._create_radiobutton_group(
            "Entry Method:",
            ["Manual Entry", "Image-Based Entry"],
            row=1
        )
        self.substrate_choice = self._create_radiobutton_group(
            "Substrate Type:",
            self.SUBSTRATE_TYPES,
            row=2,
            default='Float'
        )

        # Manual entry fields
        self.manual_entries = []
        self.background_rgb_entries = self._create_rgb_entries("Background RGB:", row_start=4)
        self.goldflake_rgb_entries = self._create_rgb_entries("Gold Flake RGB:", row_start=8)

        # Image-based entry labels
        self.background_rgb_label = self._create_rgb_label("Background RGB:", row_start=4)
        self.goldflake_rgb_label = self._create_rgb_label("Gold Flake RGB:", row_start=8)

        # Image selection buttons
        self.background_image_button = self._create_button(
            "Select Background Image", self._select_background_image, row=3, col=0
        )
        self.goldflake_image_button = self._create_button(
            "Select Gold Flake Image", self._select_gold_flake_image, row=3, col=1
        )

        # Thickness entry
        self.thickness_label = Label(self.root, text="Enter Thickness [nm]:")
        self.thickness_entry = Entry(self.root)

        # Action buttons
        self._create_button("Process", self._process_in_thread, row=13, col=0, colspan=3)
        self._create_button("Reset", self._reset_application, row=14, col=0, colspan=3)
        self._create_button("Remap Results", self._remap_results, row=15, col=0, colspan=3)

    def _create_radiobutton_group(
        self, label_text: str, options: list, row: int, default: Optional[str] = None
    ) -> StringVar:
        """Create a labeled group of radio buttons."""
        Label(self.root, text=label_text).grid(row=row, column=0, padx=10, pady=10)
        var = StringVar(self.root)
        var.set(default if default else options[0])

        for i, option in enumerate(options):
            Radiobutton(
                self.root, text=option, variable=var, value=option
            ).grid(row=row, column=i + 1, padx=10, pady=10)

        var.trace_add('write', self._reset_on_mode_switch)
        return var

    def _create_rgb_entries(self, label_text: str, row_start: int) -> Dict[str, Entry]:
        """Create RGB value entry fields."""
        Label(self.root, text=label_text).grid(
            row=row_start, column=0, padx=10, pady=5, sticky="W"
        )
        entries = {color: Entry(self.root) for color in ["R", "G", "B"]}

        for i, (color, entry) in enumerate(entries.items()):
            entry.grid(row=row_start + 1, column=i + 1, padx=5, pady=5)
            self.manual_entries.append(entry)

        return entries

    def _create_rgb_label(self, label_text: str, row_start: int) -> Label:
        """Create a label for displaying RGB values."""
        Label(self.root, text=label_text).grid(
            row=row_start, column=0, padx=10, pady=5, sticky="W"
        )
        label = Label(self.root, text="R: , G: , B: ", anchor="w")
        label.grid(row=row_start + 1, column=0, padx=5, pady=5, sticky="W", columnspan=3)
        return label

    def _create_button(
        self, text: str, command, row: int, col: int, colspan: int = 1
    ) -> Button:
        """Create and place a button widget."""
        button = Button(self.root, text=text, command=command)
        button.grid(row=row, column=col, padx=10, pady=10, columnspan=colspan)
        return button

    # -------------------------------------------------------------------------
    # UI State Management
    # -------------------------------------------------------------------------

    def _reset_on_mode_switch(self, *args):
        """Reset application when mode or substrate changes."""
        self._reset_application(
            message="The application has been reset due to mode/substrate change."
        )

    def _reset_application(self, message: str = "The application has been reset."):
        """Reset all application state."""
        self.background_image_loaded = False
        self.goldflake_image_loaded = False
        self.background_rgb_label.config(text="R: , G: , B: ")
        self.goldflake_rgb_label.config(text="R: , G: , B: ")

        for entry in self.manual_entries:
            entry.delete(0, "end")

        self._update_entry_options()
        self.root.title("Lookup Table Creation / Color to Thickness Mapping")
        messagebox.showinfo("Reset", message)

    def _update_entry_options(self, *args):
        """Update UI based on selected entry method."""
        entry_type = self.entry_choice.get()
        function_type = self.function_choice.get()

        if entry_type == "Manual Entry":
            self.background_image_button.grid_remove()
            self.goldflake_image_button.grid_remove()
            self._toggle_manual_entry_fields(show=True)

            if function_type == "Lookup Table Entry":
                self.thickness_label.grid(row=12, column=0, padx=10, pady=10)
                self.thickness_entry.grid(row=12, column=1, padx=10, pady=10)
            else:
                self._toggle_thickness_entry(show=False)
        else:
            self._toggle_manual_entry_fields(show=False)
            self.background_rgb_label.grid()
            self.goldflake_rgb_label.grid()
            self.background_image_button.grid(row=3, column=0, padx=10, pady=10)
            self.goldflake_image_button.grid(row=3, column=1, padx=10, pady=10)

            if function_type == "Lookup Table Entry":
                self.thickness_label.grid(row=12, column=0, padx=10, pady=10)
                self.thickness_entry.grid(row=12, column=1, padx=10, pady=10)
            else:
                self._toggle_thickness_entry(show=False)

    def _toggle_manual_entry_fields(self, show: bool):
        """Show or hide manual RGB entry fields."""
        for entry in self.manual_entries:
            if show:
                entry.grid()
            else:
                entry.grid_remove()

        if not show:
            self.background_rgb_label.grid_remove()
            self.goldflake_rgb_label.grid_remove()

    def _toggle_thickness_entry(self, show: bool):
        """Show or hide thickness entry field."""
        if show:
            self.thickness_label.grid()
            self.thickness_entry.grid()
        else:
            self.thickness_label.grid_remove()
            self.thickness_entry.grid_remove()

    # -------------------------------------------------------------------------
    # Image Selection and Processing
    # -------------------------------------------------------------------------

    def _select_background_image(self):
        """Launch background image selection in a thread."""
        threading.Thread(target=self._load_background_image).start()

    def _load_background_image(self):
        """Load and process background image."""
        self.background_rgb, self.background_lab = self._get_color_values(downscale=True)
        if self.background_rgb is not None:
            self.background_image_loaded = True
            self._update_rgb_label(self.background_rgb_label, self.background_rgb)

    def _select_gold_flake_image(self):
        """Launch gold flake image selection in a thread."""
        threading.Thread(target=self._load_gold_flake_image).start()

    def _load_gold_flake_image(self):
        """Load and process gold flake image."""
        self.gold_flake_rgb, self.gold_flake_lab = self._get_color_values(downscale=True)
        if self.gold_flake_rgb is not None:
            self.goldflake_image_loaded = True
            self._update_rgb_label(self.goldflake_rgb_label, self.gold_flake_rgb)

    def _update_rgb_label(self, label: Label, rgb_values: Tuple[int, int, int]):
        """Update RGB display label with values."""
        label.config(text=f"R: {rgb_values[0]}, G: {rgb_values[1]}, B: {rgb_values[2]}")

    def _get_color_values(
        self, downscale: bool = False
    ) -> Tuple[Optional[Tuple[int, int, int]], Optional[Tuple[int, int, int]]]:
        """
        Load image, select ROI, and compute average color values.

        Args:
            downscale: Whether to downscale image for faster processing.

        Returns:
            Tuple of (rgb_values, lab_values) or (None, None) on error.
        """
        image_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not image_path:
            messagebox.showerror("Error", "No image selected.")
            return None, None

        image = cv2.imread(image_path)

        if downscale:
            image = self._resize_image(image, max_size=400)

        roi = select_roi_adjustable(image)
        if roi is None:
            messagebox.showerror("Error", "No ROI selected.")
            return None, None

        # Extract ROI
        x, y, w, h = roi
        selected_area_rgb = image[y:y + h, x:x + w]
        selected_area_lab = cv2.cvtColor(selected_area_rgb, cv2.COLOR_BGR2LAB)

        # Compute average colors
        rgb = tuple(map(int, np.mean(selected_area_rgb, axis=(0, 1))))
        lab = tuple(map(int, np.mean(selected_area_lab, axis=(0, 1))))

        self.selected_image_name = os.path.splitext(os.path.basename(image_path))[0]

        return rgb, lab

    def _resize_image(self, image: np.ndarray, max_size: int = 800) -> np.ndarray:
        """Resize image to fit within max_size while preserving aspect ratio."""
        height, width = image.shape[:2]
        if max(width, height) > max_size:
            scale = max_size / max(width, height)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))
        return image

    def _get_rgb_from_entries(
        self, entries: Dict[str, Entry]
    ) -> Optional[Tuple[int, int, int]]:
        """Extract RGB values from manual entry fields."""
        try:
            r = int(entries["R"].get())
            g = int(entries["G"].get())
            b = int(entries["B"].get())

            if not all(0 <= value <= 255 for value in [r, g, b]):
                raise ValueError("RGB values must be between 0 and 255.")

            return r, g, b
        except ValueError as e:
            messagebox.showerror("Error", f"Please enter valid RGB values (0-255): {e}")
            return None

    # -------------------------------------------------------------------------
    # Core Processing Functions
    # -------------------------------------------------------------------------

    def _process_in_thread(self):
        """Launch processing in a separate thread."""
        threading.Thread(target=self._process).start()

    def _process(self):
        """Main processing dispatcher."""
        try:
            if self.function_choice.get() == "Lookup Table Entry":
                self._create_lookup_table_entry()
            else:
                if self.entry_choice.get() == "Image-Based Entry":
                    if not self.background_image_loaded or not self.goldflake_image_loaded:
                        messagebox.showerror(
                            "Error",
                            "Please select both background and gold flake images."
                        )
                        return
                self._map_image_to_lookup_table()
        except Exception as e:
            messagebox.showerror("Processing Error", f"An error occurred: {e}")

    def _create_lookup_table_entry(self):
        """Add a new entry to the lookup table."""
        substrate = self.substrate_choice.get()
        file_name = f'lookup_table_{substrate}.csv'

        # Load or create lookup table
        if os.path.exists(file_name):
            lookup_table = pd.read_csv(file_name)
        else:
            lookup_table = pd.DataFrame(
                columns=['R', 'G', 'B', 'L', 'a', 'b', 'Thickness [nm]']
            )

        # Get color values
        if self.entry_choice.get() == "Manual Entry":
            background_rgb = self._get_rgb_from_entries(self.background_rgb_entries)
            gold_flake_rgb = self._get_rgb_from_entries(self.goldflake_rgb_entries)
            if background_rgb is None or gold_flake_rgb is None:
                return
            background_lab = rgb_to_lab(background_rgb)
            gold_flake_lab = rgb_to_lab(gold_flake_rgb)
            normalized_rgb, normalized_lab = normalize_colors(
                gold_flake_rgb, gold_flake_lab, background_rgb, background_lab
            )
        else:
            normalized_rgb, normalized_lab = normalize_colors(
                self.gold_flake_rgb, self.gold_flake_lab,
                self.background_rgb, self.background_lab
            )

        if normalized_rgb is None:
            return

        # Get thickness
        try:
            thickness = float(self.thickness_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid thickness value.")
            return

        # Create new entry
        new_entry = pd.DataFrame({
            'R': [normalized_rgb[0]],
            'G': [normalized_rgb[1]],
            'B': [normalized_rgb[2]],
            'L': [normalized_lab[0]],
            'a': [normalized_lab[1]],
            'b': [normalized_lab[2]],
            'Thickness [nm]': [thickness]
        })

        # Append and save
        lookup_table = pd.concat([lookup_table, new_entry], ignore_index=True)
        lookup_table.to_csv(file_name, index=False)
        messagebox.showinfo("Saved", f"Lookup table saved to '{file_name}'.")

    def _map_image_to_lookup_table(self):
        """Map sample colors to thickness using lookup table."""
        # Get color values
        if self.entry_choice.get() == "Manual Entry":
            background_rgb = self._get_rgb_from_entries(self.background_rgb_entries)
            gold_flake_rgb = self._get_rgb_from_entries(self.goldflake_rgb_entries)
            if background_rgb is None or gold_flake_rgb is None:
                return
            background_lab = rgb_to_lab(background_rgb)
            gold_flake_lab = rgb_to_lab(gold_flake_rgb)
        else:
            background_rgb = self.background_rgb
            background_lab = self.background_lab
            gold_flake_rgb = self.gold_flake_rgb
            gold_flake_lab = self.gold_flake_lab

        # Load lookup table
        substrate = self.substrate_choice.get()
        file_name = f'lookup_table_{substrate}.csv'

        if not os.path.exists(file_name):
            messagebox.showerror(
                "Error",
                f"Lookup table for substrate '{substrate}' not found."
            )
            return

        lookup_table = pd.read_csv(file_name)

        # Normalize and find closest match
        normalized_rgb, normalized_lab = normalize_colors(
            gold_flake_rgb, gold_flake_lab, background_rgb, background_lab
        )

        thickness_rgb, match_type_rgb = self._find_closest_color(
            normalized_rgb, lookup_table, 'RGB'
        )
        thickness_lab, match_type_lab = self._find_closest_color(
            normalized_lab, lookup_table, 'LAB'
        )

        image_name = (
            self.selected_image_name
            if self.entry_choice.get() == "Image-Based Entry"
            else "Manual Entry"
        )

        # Save results
        results = pd.DataFrame({
            'Image': [image_name],
            'Substrate': [substrate],
            'Average_RGB': [normalized_rgb],
            'Average_LAB': [normalized_lab],
            'Thickness_RGB [nm]': [thickness_rgb],
            'Thickness_LAB [nm]': [thickness_lab],
            'Note_RGB': [match_type_rgb],
            'Note_LAB': [match_type_lab]
        })

        results_file = 'results.csv'
        results.to_csv(
            results_file,
            mode='a' if os.path.exists(results_file) else 'w',
            header=not os.path.exists(results_file),
            index=False
        )
        messagebox.showinfo("Results", f"Results saved to '{results_file}'.")

        # Display results
        results_text = (
            f"Normalized RGB values: {normalized_rgb}\n"
            f"Normalized LAB values: {normalized_lab}\n"
            f"Mapped thickness (RGB): {thickness_rgb} nm ({match_type_rgb})\n"
            f"Mapped thickness (LAB): {thickness_lab} nm ({match_type_lab})"
        )
        messagebox.showinfo("Results", results_text)

    def _find_closest_color(
        self,
        color_value: Tuple[float, ...],
        lookup_table: pd.DataFrame,
        color_space: str = 'RGB',
        threshold: float = 5.0
    ) -> Tuple[str, str]:
        """
        Find the closest color match in the lookup table.

        Args:
            color_value: Color values to match.
            lookup_table: DataFrame containing lookup table entries.
            color_space: 'RGB' or 'LAB' color space.
            threshold: Maximum distance for approximate match.

        Returns:
            Tuple of (thickness_string, match_type).
        """
        try:
            if color_space == 'RGB':
                diff = lookup_table[['R', 'G', 'B']] - color_value
            else:
                diff = lookup_table[['L', 'a', 'b']] - color_value

            distance = np.sqrt((diff**2).sum(axis=1))
            min_distance = distance.min()

            # Find all closest matches
            closest_indices = distance[distance == min_distance].index
            thicknesses = lookup_table.loc[closest_indices, 'Thickness [nm]']
            thickness_list = thicknesses.astype(str).tolist()

            # Determine match type
            exact_match = False
            if len(closest_indices) == 1:
                closest_diff = diff.loc[closest_indices].iloc[0]
                exact_match = (closest_diff == 0).all()

            if len(thickness_list) > 1:
                match_type = "Multiple matches"
            elif exact_match:
                match_type = "Exact match"
            elif min_distance <= threshold:
                match_type = "Approximate match"
            else:
                match_type = "No match"

            thickness_str = "/".join(thickness_list)
            return thickness_str, match_type

        except Exception as e:
            messagebox.showerror("Error", f"Failed to find closest color: {e}")
            return "", "Error"

    def _remap_results(self):
        """Remap existing results with updated lookup tables."""
        try:
            results_file = 'results.csv'
            if not os.path.exists(results_file):
                messagebox.showerror(
                    "Error",
                    "No results file found. Please generate results first."
                )
                return

            results = pd.read_csv(results_file)
            updated_results = []

            for _, row in results.iterrows():
                # Parse stored color values
                normalized_rgb = tuple(
                    map(float, row['Average_RGB'].strip('()').split(','))
                )
                normalized_lab = tuple(
                    map(float, row['Average_LAB'].strip('()').split(','))
                )

                substrate = row['Substrate']
                lookup_table_file = f'lookup_table_{substrate}.csv'

                if not os.path.exists(lookup_table_file):
                    messagebox.showerror(
                        "Error",
                        f"Lookup table for substrate '{substrate}' not found."
                    )
                    return

                lookup_table = pd.read_csv(lookup_table_file)

                # Remap
                thickness_rgb, match_type_rgb = self._find_closest_color(
                    normalized_rgb, lookup_table, 'RGB'
                )
                thickness_lab, match_type_lab = self._find_closest_color(
                    normalized_lab, lookup_table, 'LAB'
                )

                # Update row
                updated_row = row.copy()
                updated_row['Thickness_RGB [nm]'] = thickness_rgb
                updated_row['Thickness_LAB [nm]'] = thickness_lab
                updated_row['Note_RGB'] = match_type_rgb
                updated_row['Note_LAB'] = match_type_lab
                updated_results.append(updated_row)

            # Save updated results
            final_results = pd.DataFrame(updated_results).drop_duplicates()
            final_results.to_csv(results_file, index=False)
            messagebox.showinfo("Success", "Results have been remapped and consolidated.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to remap results: {e}")

    def _on_closing(self):
        """Handle application close event."""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.root.destroy()


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Launch the Color to Thickness mapping application."""
    root = Tk()
    app = ColorToThicknessApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
