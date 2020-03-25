# Download inria dataset
wget -q --no-check-certificate https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.001
wget -q --no-check-certificate https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.002
wget -q --no-check-certificate https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.003
wget -q --no-check-certificate https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.004
wget -q --no-check-certificate https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.005

7zr x aerialimagelabeling.7z.001 -tsplit
rm aerialimagelabeling.7z.* 
p7zip -d aerialimagelabeling.7z
unzip NEW2-AerialImageDataset.zip
rm NEW2-AerialImageDataset.zip