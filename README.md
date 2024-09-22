# CLIP-Image-Search

A simple image search system using the CLIP (Contrastive Language-Image Pre-training) model. This repository contains code for embedding images and text, and performing similarity searches based on text queries.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/babulaaldon/CLIP-Model-Implementation.git
    cd CLIP-Image-Search
    ```

2. Install the required packages:
    ```bash
    pip install torch clip pillow numpy scikit-learn
    ```

## Usage

1. Update the `image_path` variable in the script to point to your image file.
2. Run the script:
    ```bash
    python img_sea.py
    ```

## Code Structure

- `main.py`: Contains the main code for the image search system.
- `imgs/`: Directory containing image files.

## Dependencies

- [PyTorch](https://pytorch.org/)
- [CLIP](https://github.com/openai/CLIP)
- [Pillow](https://pillow.readthedocs.io/)
- [NumPy](https://numpy.org/)
- [Scikit-learn](https://scikit-learn.org/)

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
