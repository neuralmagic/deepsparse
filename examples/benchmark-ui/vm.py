import digitalocean
import subprocess


class CPUHandler:
    
    def get_cpu_model_name(self):
  
        cmd = ["cat /proc/cpuinfo | grep 'model name' | uniq"]
        output = subprocess.check_output(cmd, shell=True).decode("utf-8")
        
        return output.replace("model name", "CPU Model ").strip()
        
    def get_cpu_count(self):
        
        cmd = ["nproc"]
        output = subprocess.check_output(cmd, shell=True).decode("utf-8")
        
        return f"Total CPUs: {output}"

        

class Droplet:

    def __init__(
        self, 
        token: str, 
        name: str, 
        region: str,
        image: str, 
        size_slug:str, 
        backups: bool
    ) -> None:
        
        self.token=token
        self.name=name
        self.region=region
        self.image=image
        self.size_slug=size_slug
        self.backups=backups
      
        self.manager = digitalocean.Manager(token=self.token)
        self.ssh_keys = self.manager.get_all_sshkeys()
        
        self.droplet = digitalocean.Droplet(
            token=self.token,
            name=self.name,
            region=self.region,
            ssh_keys=self.ssh_keys,
            image=self.image,
            size_slug=self.size_slug,
            backups=self.backups
        )

    def create_droplet(self):

        self.droplet.create()
        print("droplet is staging...")