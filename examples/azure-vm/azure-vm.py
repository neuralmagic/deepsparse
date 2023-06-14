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

import base64

import click

from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.resource import ResourceManagementClient


def create_resource_group(resource_client: str, group_name: str, location: str):
    resource_client.resource_groups.create_or_update(group_name, {"location": location})


def create_virtual_network(
    network_client: str, group_name: str, network_name: str, location: str
):
    network_client.virtual_networks.begin_create_or_update(
        group_name,
        network_name,
        {"location": location, "address_space": {"address_prefixes": ["10.0.0.0/16"]}},
    ).result()


def create_subnet(
    network_client: str, group_name: str, network_name: str, subnet_name: str
):
    return network_client.subnets.begin_create_or_update(
        group_name, network_name, subnet_name, {"address_prefix": "10.0.0.0/24"}
    ).result()


def create_network_security_group(
    network_client: str, group_name: str, nsg_name: str, location: str
):
    nsg_params = {
        "location": location,
        "security_rules": [
            {
                "name": "SSH",
                "protocol": "Tcp",
                "source_port_range": "*",
                "destination_port_range": "22",
                "source_address_prefix": "*",
                "destination_address_prefix": "*",
                "access": "Allow",
                "priority": 100,
                "direction": "Inbound",
            }
        ],
    }
    return network_client.network_security_groups.begin_create_or_update(
        group_name, nsg_name, nsg_params
    ).result()


def create_public_ip_address(
    network_client: str, group_name: str, ip_address_name: str, location: str
):
    public_ip_address_params = {
        "location": location,
        "public_ip_allocation_method": "static",
        "public_ip_address_version": "ipv4",
    }
    return network_client.public_ip_addresses.begin_create_or_update(
        group_name, ip_address_name, public_ip_address_params
    ).result()


def create_network_interface(
    network_client: str,
    group_name: str,
    interface_name: str,
    location: str,
    subnet: str,
    public_ip_address: str,
    nsg: str,
):
    network_interface_params = {
        "location": location,
        "ip_configurations": [
            {
                "name": "MyIpConfig",
                "subnet": {"id": subnet.id},
                "public_ip_address": {"id": public_ip_address.id},
            }
        ],
        "network_security_group": {"id": nsg.id},
    }
    return network_client.network_interfaces.begin_create_or_update(
        group_name, interface_name, network_interface_params
    ).result()


def create_virtual_machine(
    compute_client: str,
    group_name: str,
    vm_name: str,
    location: str,
    vm_type: str,
    interface_name: str,
    subscription_id: str,
    PASSWORD: str,
    startup_script: str,
):
    return compute_client.virtual_machines.begin_create_or_update(
        group_name,
        vm_name,
        {
            "location": location,
            "hardware_profile": {"vm_size": vm_type},
            "storage_profile": {
                "image_reference": {
                    "sku": "20_04-lts-gen2",
                    "publisher": "Canonical",
                    "version": "latest",
                    "offer": "0001-com-ubuntu-server-focal",
                },
                "os_disk": {
                    "caching": "ReadWrite",
                    "managed_disk": {"storage_account_type": "Standard_LRS"},
                    "name": "myVMosdisk",
                    "create_option": "FromImage",
                },
                "data_disks": [
                    {"disk_size_gb": "1023", "create_option": "Empty", "lun": "0"},
                    {"disk_size_gb": "1023", "create_option": "Empty", "lun": "1"},
                ],
            },
            "os_profile": {
                "admin_username": "testuser",
                "computer_name": "myVM",
                "admin_password": PASSWORD,
                "custom_data": startup_script,
            },
            "network_profile": {
                "network_interfaces": [
                    {
                        "id": "/subscriptions/"
                        + subscription_id
                        + "/resourceGroups/"
                        + group_name
                        + "/providers/Microsoft.Network/networkInterfaces/"
                        + interface_name
                        + "",
                        "properties": {"primary": True},
                    }
                ]
            },
        },
    ).result()


@click.group()
def cli():
    pass


@cli.command()
@click.option("--subscription-id", required=True, help="Azure subscription ID")
@click.option("--location", required=True, help="Location")
@click.option("--vm-type", required=True, help="Virtual machine type")
@click.option("--group-name", required=True, help="Resource group name")
@click.option("--vm-name", required=True, help="Virtual machine name")
@click.option("--pw", required=True, help="Virtual machine password")
def create_vm(
    subscription_id: str,
    location: str,
    vm_type: str,
    group_name: str,
    vm_name: str,
    pw: str,
):
    """
    Create a new virtual machine in Azure.

    Args:
        subscription_id (str): Azure subscription ID.
        location (str): Location where the VM will be created.
        vm_type (str): Type of virtual machine (size).
        group_name (str): Name of the resource group to create.
        vm_name (str): Name of the virtual machine to create.
        pw (str): Password for the virtual machine.
    """
    SUBSCRIPTION_ID = subscription_id
    LOCATION = location
    VM_TYPE = vm_type
    GROUP_NAME = group_name
    VIRTUAL_MACHINE_NAME = vm_name
    PASSWORD = pw
    SUBNET_NAME = "subnetx"
    INTERFACE_NAME = "interfacex"
    NETWORK_NAME = "networknamex"
    IP_ADDRESS_NAME = "ipaddressx"
    NSG_NAME = "nsgx"

    startup_script = """#!/bin/bash
    apt-get update
    apt-get install -y apt-transport-https ca-certificates curl \\
        software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \\
        sudo apt-key add -
    add-apt-repository "deb [arch=amd64] \\
        https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
    apt-get update
    apt-get install -y docker-ce
    docker pull ghcr.io/neuralmagic/deepsparse:1.4.2
    docker tag ghcr.io/neuralmagic/deepsparse:1.4.2 deepsparse_docker
    """

    # Encode startup_script as Base64
    startup_script = base64.b64encode(startup_script.encode()).decode()

    # Create client
    resource_client = ResourceManagementClient(
        credential=DefaultAzureCredential(), subscription_id=SUBSCRIPTION_ID
    )
    network_client = NetworkManagementClient(
        credential=DefaultAzureCredential(), subscription_id=SUBSCRIPTION_ID
    )
    compute_client = ComputeManagementClient(
        credential=DefaultAzureCredential(), subscription_id=SUBSCRIPTION_ID
    )

    create_resource_group(resource_client, GROUP_NAME, LOCATION)
    create_virtual_network(network_client, GROUP_NAME, NETWORK_NAME, LOCATION)
    subnet = create_subnet(network_client, GROUP_NAME, NETWORK_NAME, SUBNET_NAME)
    nsg = create_network_security_group(network_client, GROUP_NAME, NSG_NAME, LOCATION)
    public_ip_address = create_public_ip_address(
        network_client, GROUP_NAME, IP_ADDRESS_NAME, LOCATION
    )
    create_network_interface(
        network_client,
        GROUP_NAME,
        INTERFACE_NAME,
        LOCATION,
        subnet,
        public_ip_address,
        nsg,
    )
    create_virtual_machine(
        compute_client,
        GROUP_NAME,
        VIRTUAL_MACHINE_NAME,
        LOCATION,
        VM_TYPE,
        INTERFACE_NAME,
        SUBSCRIPTION_ID,
        PASSWORD,
        startup_script,
    )

    print("Your external public IP address:", public_ip_address.ip_address)


@cli.command()
@click.option("--subscription-id", required=True, help="Azure subscription ID")
@click.option("--group-name", required=True, help="Resource group name")
@click.option("--vm-name", required=True, help="Virtual machine name")
def delete_vm_rg(subscription_id: str, group_name: str, vm_name: str):
    compute_client = ComputeManagementClient(
        credential=DefaultAzureCredential(), subscription_id=subscription_id
    )
    resource_client = ResourceManagementClient(
        credential=DefaultAzureCredential(), subscription_id=subscription_id
    )

    compute_client.virtual_machines.begin_power_off(group_name, vm_name).result()
    compute_client.virtual_machines.begin_delete(group_name, vm_name).result()
    print("Deleted virtual machine.")

    resource_client.resource_groups.begin_delete(group_name).result()
    print("Deleted resource group.")


if __name__ == "__main__":
    cli()
