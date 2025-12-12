pip install "numpy<1.20"
pip install 'pillow<10'
pip install torch==2.8.0+cu128 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
pip install chainercv