class TreeNode:
    def __init__(self, data, completion_status="todo"):
        self.data = data
        assert completion_status in ["todo", "inprogress", "done"]
        self.completion_status = completion_status
        self.children = []
    def add(self, child):
        if isinstance(child, TreeNode):
            self.children.append(child)
        else:
            print("Child must be an instance of TreeNode")
    def remove(self, child):
        if child in self.children:
            self.children.remove(child)
        else:
            print("Child not found")
    def __str__(self, level=-1, prefix=""):
        if len(prefix) > 0:
            ret = "\t" * level + prefix + ": " + self.data + f"-({self.completion_status})" + "\n"
        else:
            ret = ""

        for i, child in enumerate(self.children):
            ret += child.__str__(level + 1, f"{prefix}.{i+1}" if len(prefix) > 0 else f"{i+1}")
        return ret

if __name__ == "__main__":
    # Create the root of the PTT
    root = TreeNode("Pentesting")

    # Phase 1: Reconnaissance
    recon = TreeNode("Reconnaissance")

    # Passive Information Gathering (Completed)
    passive_info_gathering = TreeNode("Passive Information Gathering", "done")

    # Active Information Gathering (Completed)
    active_info_gathering = TreeNode("Active Information Gathering", "done")

    # Identify Open Ports and Services
    identify_open_ports = TreeNode("Identify Open Ports and Services")

    # Perform a full port scan
    identify_open_ports.add(TreeNode("Perform a full port scan"))

    # Determine the purpose of each open port
    identify_open_ports.add(TreeNode("Determine the purpose of each open port"))

    recon.add(passive_info_gathering)
    recon.add(active_info_gathering)
    recon.add(identify_open_ports)

    # Phase 2: Enumeration
    enumeration = TreeNode("Enumeration")

    # Enumerate Services on IP 10.23.42.434
    enumerate_services = TreeNode("Enumerate Services on IP 10.23.42.434")
    enumerate_services.add(TreeNode("Identify running services"))
    enumerate_services.add(TreeNode("Version detection for services"))

    # Enumerate Users and Shares on the Target
    enumerate_users_shares = TreeNode("Enumerate Users and Shares on the Target")
    enumerate_users_shares.add(TreeNode("Identify user accounts"))
    enumerate_users_shares.add(TreeNode("Identify shared resources"))

    enumeration.add(enumerate_services)
    enumeration.add(enumerate_users_shares)

    # Phase 3: Vulnerability Scanning (if not already done)
    vulnerability_scanning = TreeNode("Vulnerability Scanning")
    vulnerability_scanning.add(TreeNode("Perform vulnerability scanning"))

    # Phase 4: Exploitation (if vulnerabilities found)
    exploitation = TreeNode("Exploitation")
    exploitation.add(TreeNode("Exploit identified vulnerabilities"))

    # Phase 5: Post-Exploitation (Not to be generated in this plan)
    # Skip post-exploitation in the simulation environment as requested.

    root.add(recon)
    root.add(enumeration)
    root.add(vulnerability_scanning)
    root.add(exploitation)
    print(root)

