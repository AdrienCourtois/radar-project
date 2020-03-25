# Download Paris dataset
wget -q --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1sPiJTuJvoEeQk4IIh2n6QuOPg5FPaAtv' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1sPiJTuJvoEeQk4IIh2n6QuOPg5FPaAtv" -O SN2_buildings_train_AOI_3_Paris.tar.gz && rm -rf /tmp/cookies.txt
# Extract
tar -xzf SN2_buildings_train_AOI_3_Paris.tar.gz
# Delete
rm SN2_buildings_train_AOI_3_Paris.tar.gz