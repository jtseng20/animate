![logo](/assets/logo.png "anim∞")

## anim∞ *[pron. animate]*
This is the repository for the portable demo notebook for anim∞, a project made for HackMIT 2022.

### How to use
Just run the notebook! The notebook contains a basic outline for class-conditional animation generation and a tiny and pretty-much-universally supported ONNX build of the Temporal U-Net that drives our motion model. This model is a compressed version that should run on any machine.

### Installation Notes
- Please download the model checkpoint and related data from the Google Drive folder at https://drive.google.com/drive/folders/1MPHu2Odahb8V5C9Nw8jRp56vH-pqoZ3a?usp=sharing
- When installing `onnxruntime`, the capabilities of your local machine may necessitate installation of a CPU/GPU-exclusive version. See https://onnxruntime.ai/docs/install/. 