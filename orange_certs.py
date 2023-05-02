import sys
import os
try:
    import requests
    import certifi
except ImportError:
    print("Python certifi not installed ignoring certificate installation")
    sys.exit(0)

certs_base_url = "http://web2000.rd.francetelecom.fr/certificats.dir/certifs/format_pem.dir/"

orange_certs = [
    "Groupe_France_Telecom_Root_CA.pem",
    "Groupe_France_Telecom_Root_CA_1-2.pem",
    "Groupe_France_Telecom_Root_CA_1-3.pem",
    "Groupe_France_Telecom_Root_CA_1.pem",
    "Groupe_France_Telecom_Internal_CA1-4.pem",
    "Orange_Internal_G2_Root_CA.pem",
]

cafile = certifi.where()
with open(cafile,"r",encoding='utf-8') as f:
    ca_bundle = f.read()

for ca_name in orange_certs:
    print("Downloading certificate: %s"%ca_name)
    ca_data = requests.get(certs_base_url+ca_name).content
    if ca_data.decode("utf-8") in ca_bundle:
        print("Skipping %s"%ca_name)
        continue
    print("Adding %s"%ca_name)
    with open(cafile, "ab") as outfile:
        outfile.write(os.linesep.encode("utf-8"))
        outfile.write(ca_data)
