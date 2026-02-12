DATASETS_DIR=$HOME/datasets

# 2D data
wget https://surfer.nmr.mgh.harvard.edu/ftp/data/neurite/data/neurite-oasis.2d.v1.0.tar
mkdir -p "${DATASETS_DIR}/OASIS-2d"
tar xf neurite-oasis.2d.v1.0.tar --directory "${DATASETS_DIR}/OASIS-2d"
rm neurite-oasis.2d.v1.0.tar

# 3D data
wget https://surfer.nmr.mgh.harvard.edu/ftp/data/neurite/data/neurite-oasis.v1.0.tar
mkdir -p "${DATASETS_DIR}/OASIS"
tar xf neurite-oasis.v1.0.tar --directory "${DATASETS_DIR}/OASIS"
rm neurite-oasis.v1.0.tar
