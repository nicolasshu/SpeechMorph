mkdir datasets
cd datasets 
wget http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_awb_arctic-0.95-release.zip
wget http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_bdl_arctic-0.95-release.zip
wget http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_clb_arctic-0.95-release.zip
wget http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_jmk_arctic-0.95-release.zip
wget http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_ksp_arctic-0.95-release.zip
wget http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_rms_arctic-0.95-release.zip
wget http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_slt_arctic-0.95-release.zip


mkdir awb
mkdir bdl
mkdir clb 
mkdir jmk
mkdir ksp
mkdir rms
mkdir slt

unzip cmu_us_awb_arctic-0.95-release.zip -d awb
unzip cmu_us_bdl_arctic-0.95-release.zip -d bld
unzip cmu_us_clb_arctic-0.95-release.zip -d clb
unzip cmu_us_jmk_arctic-0.95-release.zip -d jmk
unzip cmu_us_ksp_arctic-0.95-release.zip -d ksp
unzip cmu_us_rms_arctic-0.95-release.zip -d rms
unzip cmu_us_slt_arctic-0.95-release.zip -d slt

