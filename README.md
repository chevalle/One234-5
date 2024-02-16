**Pre-requisite :**

```
Ubuntu, 22.04 LTS (latest, first one in the list)
Instance type: g5.2xlarge instance
Storage : 200Gb
Security group : SSH (if you use that), 7860 port allowed 
```

**IAM role associated to instance with :**

- ECR-public access to download container image
- sts:GetServiceBearerToken allowed
- Bedrock invoke model (optional, you can import your own images)


**1 - Connect to instance, clone repo and start runbook to install dependencies and download container**

```
git clone https://github.com/chevalle/One234-5.git
cd One234-5
chmod +x runbook.sh
./runbook.sh
```

**2 - During installation press enter if you are prompted to do something**

**3 - The script will reboot the instance when done**

**4 - Reconnect, run the container.**

```
sudo docker run -p 7860:7860 --name One-2-3-45_demo --gpus all -it public.ecr.aws/f8j2d1v7/one2345-3dprint
```

**5 - Connect to http://{ec2-public-ip}:7860 to use the gradio interface**

**API reference :**

**To begin, initialize the Gradio Client with the API URL.**

```
from gradio_client import Client
client = Client("https://one-2-3-45-one-2-3-45.hf.space/")
# example input image
input_img_path = "https://huggingface.co/spaces/One-2-3-45/One-2-3-45/resolve/main/demo_examples/01_wild_hydrant.png"
```

**Single image to 3D mesh**

```
generated_mesh_filepath = client.predict(
	input_img_path,	
	True,		# image preprocessing
	api_name="/generate_mesh"
)
```

**Elevation estimation**
If the input image's pose (elevation) is unknown, this off-the-shelf algorithm is all you need!

```
elevation_angle_deg = client.predict(
	input_img_path,
	True,		# image preprocessing
	api_name="/estimate_elevation"
)
```

**Image preprocessing: segment, rescale, and recenter**
We adapt the Segment Anything model (SAM) for background removal.

```
segmented_img_filepath = client.predict(
	input_img_path,	
	api_name="/preprocess"
)
```
