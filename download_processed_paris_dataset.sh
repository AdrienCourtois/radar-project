
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1rmMUXXjL5V02Cs0iKv7_LESP7F8FgPDe' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1rmMUXXjL5V02Cs0iKv7_LESP7F8FgPDe" -O RGB-normalized.tar.gz && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-1Bw19WI6U406Ycgw9A6IQgJj-Dg30o-' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-1Bw19WI6U406Ycgw9A6IQgJj-Dg30o-" -O segmentation.tar.gz && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-85FrXBLhdHGx_W3qlYCGktPoXRJJJ2A' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-85FrXBLhdHGx_W3qlYCGktPoXRJJJ2A" -O MUL-normalized.tar.gz && rm -rf /tmp/cookies.txt

# Extract
tar -xzf RGB-normalized.tar.gz
tar -xzf segmentation.tar.gz
tar -xzf MUL-normalized.tar.gz
# Delete
rm RGB-normalized.tar.gz
rm segmentation.tar.gz
rm MUL-normalized.tar.gz