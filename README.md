Pre-requisite :

```
Ubuntu, 22.04 LTS (latest, first one in the list)
Instance type: g5.2xlarge instance
Storage : 200Gb
Security group : SSH (if you use that), 7860 port allowed 
```

IAM role associated to instance with : 

- ECR-public access to download container image
- sts:GetServiceBearerToken allowed
- Bedrock invoke model (optional, you can import your own images)


1 - Connect to instance, clone repo and start runbook to install dependencies and download container 

```
git clone https://github.com/chevalle/One234-5.git
cd One234-5
chmod +x runbook.sh
./runbook.sh
```

2 - During installation press enter if you are prompted to do something 

3 - The script will reboot the instance

4 - Run the container 

```
sudo docker run -p 7860:7860 --name One-2-3-45_demo --gpus all -it public.ecr.aws/f8j2d1v7/one2345-3dprint
```
