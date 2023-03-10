# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import psutil
import subprocess

import digitalocean


class CPUHandler:
    
    def get_cpu_count(self):

        return f"Total CPUs: {psutil.cpu_count()}"
    
    def get_cpu_model_name(self):

        cmd = ["cat /proc/cpuinfo | grep 'model name' | uniq"]
        output = subprocess.check_output(cmd, shell=True).decode("utf-8")

        return output.replace("model name", "CPU Model ").strip()
    
    def get_ram(self):
        
        total_ram = psutil.virtual_memory().total
        total_ram = psutil._common.bytes2human(total_ram)

        return f"Total RAM: {total_ram}"


class Droplet:
    def __init__(
        self,
        token: str,
        name: str,
        region: str,
        image: str,
        size_slug: str,
        backups: bool,
    ) -> None:

        self.token = token
        self.name = name
        self.region = region
        self.image = image
        self.size_slug = size_slug
        self.backups = backups

        self.manager = digitalocean.Manager(token=self.token)
        self.ssh_keys = self.manager.get_all_sshkeys()

        self.droplet = digitalocean.Droplet(
            token=self.token,
            name=self.name,
            region=self.region,
            ssh_keys=self.ssh_keys,
            image=self.image,
            size_slug=self.size_slug,
            backups=self.backups,
        )

    def create_droplet(self):

        self.droplet.create()
        print("droplet is staging...")
