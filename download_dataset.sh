wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz
mkdir -p data
cd data
wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip
unzip nerf_example_data.zip
wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
mkdir 360_v2
unzip 360_v2.zip -d 360_v2
cd ..
