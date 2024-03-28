"""
Created on Wed Oct 11 18:07:27 2023

@author: Chris

Description: A python script to parse .pcap files - in particular, the 5 
             files part of the CIC-IDS2017 dataset , for Monday - Friday.
             The script extracts features from IP, TCP, and UDP headers 
             and writes these to .csv files
             The .csv files store x amount of packets and the script 
             ensures that files are not cut in between flows
             
Version: v2
"""

import os
import csv
import dpkt
import datetime
from dpkt.utils import mac_to_str, inet_to_str


def write_to_csv(csv_file, packet_features):

    # Create a list of column headers based on the keys in your dictionary
    csv_columns = list(packet_features.keys())
    
    # Open the CSV file in write mode
    with open(csv_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=csv_columns)
        
        # Write the header row
        writer.writeheader()
        
        # Iterate through the data and write each row
        for i in range(len(packet_features['packet_num'])):
            row_data = {key: packet_features[key][i] for key in csv_columns}
            writer.writerow(row_data)
            
        
"""
 Select Weekday & Number of Packets in each file
"""

weekday = 'Thursday'

# Define folder name and path
file_name = weekday + '-WorkingHours'  
folder_path = os.path.join(os.getcwd(), file_name)                              # Path to folder to read and write files


"""
    Definition of IP Addresses of attackers and victims
"""

firewall = "172.16.0.1"

kali = "205.174.165.73"

ubuntu16 = "192.168.10.50"
ubuntu12 = "192.168.10.51"
vista = "192.168.10.8"
mac = "192.168.10.25"


win7 = "192.168.10.9"
win8 = "192.168.10.5"
win10_32B = "192.168.10.14"
win10_64B = "192.168.10.15"



if weekday == 'Tuesday':
    attacks = [
        [firewall, ubuntu16, datetime.datetime(2017,7,4,9,20,0), datetime.datetime(2017,7,4,10,20,0), "FTP" ],
        [firewall, ubuntu16, datetime.datetime(2017,7,4,14,0,0), datetime.datetime(2017,7,4,15,0,0), "SSH" ]
    ]
elif weekday == 'Wednesday':
    
    attacks = [
        
        [firewall, ubuntu16, datetime.datetime(2017,7,5,9,47,0), datetime.datetime(2017,7,5,10,10,0), "slowloris" ],
        [firewall, ubuntu16, datetime.datetime(2017,7,5,10,14,0), datetime.datetime(2017,7,5,10,35,0), "slowhttptest" ],
        [firewall, ubuntu16, datetime.datetime(2017,7,5,10,43,0), datetime.datetime(2017,7,5,11,0,0), "hulk" ],
        [firewall, ubuntu16, datetime.datetime(2017,7,5,11,10,0), datetime.datetime(2017,7,5,11,23,0), "goldeneye" ],
        [firewall, ubuntu12, datetime.datetime(2017,7,5,15,12,0), datetime.datetime(2017,7,5,15,32,0), "heartbleed" ]
    ]
    
elif weekday == 'Thursday':
    attacks = [

        
        [firewall, ubuntu16, datetime.datetime(2017,7,6,9,20,0), datetime.datetime(2017,7,6,10,0,0), "bruteforce" ],
        [firewall, ubuntu16, datetime.datetime(2017,7,6,10,15,0), datetime.datetime(2017,7,6,10,35,0), "xss" ],
        [firewall, ubuntu16, datetime.datetime(2017,7,6,10,40,0), datetime.datetime(2017,7,6,10,42,0), "sqlinject" ],
        [kali, vista, datetime.datetime(2017,7,6,14,19,0), datetime.datetime(2017,7,6,14,21,0), "infiltration" ],
        [kali, vista, datetime.datetime(2017,7,6,14,33,0), datetime.datetime(2017,7,6,14,35,0), "infiltration" ],
        [kali, mac, datetime.datetime(2017,7,6,14,53,0), datetime.datetime(2017,7,6,15,0,0), "infiltration" ],
        [kali, vista, datetime.datetime(2017,7,6,15,4,0), datetime.datetime(2017,7,6,15,45,0), "infiltration"],
        [vista, "all", datetime.datetime(2017,7,6,15,4,0), datetime.datetime(2017,7,6,15,45,0), "infiltration" ]
   
    ]
        
elif weekday == 'Friday':
    attacks = [
        
        [kali, win10_64B, datetime.datetime(2017,7,7,10,2,0), datetime.datetime(2017,7,7,11,2,0), "botnet" ],
        [kali, win10_32B, datetime.datetime(2017,7,7,10,2,0), datetime.datetime(2017,7,7,11,2,0), "botnet" ],
        [kali, win7, datetime.datetime(2017,7,7,10,2,0), datetime.datetime(2017,7,7,11,2,0), "botnet" ],
        [kali, win8, datetime.datetime(2017,7,7,10,2,0), datetime.datetime(2017,7,7,11,2,0), "botnet" ],
        [kali, vista, datetime.datetime(2017,7,7,10,2,0), datetime.datetime(2017,7,7,11,2,0), "botnet" ],

        [firewall, ubuntu16, datetime.datetime(2017,7,7,13,55,0), datetime.datetime(2017,7,7,13,57,0) , "portscan_on" ],
        [firewall, ubuntu16, datetime.datetime(2017,7,7,13,58,0), datetime.datetime(2017,7,7,14,0,0), "portscan_on" ],
        [firewall, ubuntu16, datetime.datetime(2017,7,7,14,1,0), datetime.datetime(2017,7,7,14,4,0), "portscan_on" ],
        [firewall, ubuntu16, datetime.datetime(2017,7,7,14,5,0), datetime.datetime(2017,7,7,14,7,0), "portscan_on" ],
        [firewall, ubuntu16, datetime.datetime(2017,7,7,14,8,0), datetime.datetime(2017,7,7,14,10,0), "portscan_on" ],
        [firewall, ubuntu16, datetime.datetime(2017,7,7,14,11,0), datetime.datetime(2017,7,7,14,13,0), "portscan_on" ],
        [firewall, ubuntu16, datetime.datetime(2017,7,7,14,14,0), datetime.datetime(2017,7,7,14,16,0), "portscan_on" ],
        [firewall, ubuntu16, datetime.datetime(2017,7,7,14,17,0), datetime.datetime(2017,7,7,14,19,0), "portscan_on" ],
        [firewall, ubuntu16, datetime.datetime(2017,7,7,14,20,0), datetime.datetime(2017,7,7,14,21,0), "portscan_on" ],
        [firewall, ubuntu16, datetime.datetime(2017,7,7,14,22,0), datetime.datetime(2017,7,7,14,24,0), "portscan_on" ],
        [firewall, ubuntu16, datetime.datetime(2017,7,7,14,33,0), datetime.datetime(2017,7,7,14,34,0), "portscan_on" ],
        [firewall, ubuntu16, datetime.datetime(2017,7,7,14,35,0), datetime.datetime(2017,7,7,14,36,0), "portscan_on" ],
        
        [firewall, ubuntu16, datetime.datetime(2017,7,7,14,51,0), datetime.datetime(2017,7,7,14,53,0), "portscan_off" ],
        [firewall, ubuntu16, datetime.datetime(2017,7,7,14,54,0), datetime.datetime(2017,7,7,14,56,0), "portscan_off" ],
        [firewall, ubuntu16, datetime.datetime(2017,7,7,14,57,0), datetime.datetime(2017,7,7,14,59,0), "portscan_off" ],
        [firewall, ubuntu16, datetime.datetime(2017,7,7,15,0,0), datetime.datetime(2017,7,7,15,2,0), "portscan_off" ],
        [firewall, ubuntu16, datetime.datetime(2017,7,7,15,3,0), datetime.datetime(2017,7,7,15,5,0), "portscan_off" ],
        [firewall, ubuntu16, datetime.datetime(2017,7,7,15,6,0), datetime.datetime(2017,7,7,15,7,0), "portscan_off" ],
        [firewall, ubuntu16, datetime.datetime(2017,7,7,15,8,0), datetime.datetime(2017,7,7,15,10,0), "portscan_off" ],
        [firewall, ubuntu16, datetime.datetime(2017,7,7,15,11,0), datetime.datetime(2017,7,7,15,12,0), "portscan_off" ],
        [firewall, ubuntu16, datetime.datetime(2017,7,7,15,13,0), datetime.datetime(2017,7,7,15,15,0), "portscan_off" ],
        [firewall, ubuntu16, datetime.datetime(2017,7,7,15,16,0), datetime.datetime(2017,7,7,15,18,0), "portscan_off" ],
        [firewall, ubuntu16, datetime.datetime(2017,7,7,15,19,0), datetime.datetime(2017,7,7,15,21,0), "portscan_off" ],
        [firewall, ubuntu16, datetime.datetime(2017,7,7,15,22,0), datetime.datetime(2017,7,7,15,24,0), "portscan_off" ],
        [firewall, ubuntu16, datetime.datetime(2017,7,7,15,25,0), datetime.datetime(2017,7,7,15,27,0), "portscan_off" ],
        [firewall, ubuntu16, datetime.datetime(2017,7,7,15,28,0), datetime.datetime(2017,7,7,15,29,0), "portscan_off" ],
        
        [firewall, ubuntu16, datetime.datetime(2017,7,7,15,56,0), datetime.datetime(2017,7,7,16,16,0), "DDoS" ]
    ]
    


    
"""
 Definition of lists to hold values of header features and indexing
"""

# Indexing & Timestamp
packet_num_list =  []                                                           # Index of given packet
timestamp_list =   []                                                           # Packet timestamp
datetime_list =[]                                                               # Formatted datetime
# IP header features
version_list =     []                                                           # (int): Version (4 bits) For IPv4, this is always equal to 4                     
header_len_list =  []                                                           # (int): Internet Header Length (IHL) (4 bits)
src_address_list = []                                                           # (int): Source address. This field is the IPv4 address of the sender of the packet. (4 bytes)                                                                                                   
dst_address_list = []                                                           # (int): Destination address. This field is the IPv4 address of the receiver of the packet. (4 bytes)
tot_len_list =     []                                                           # (int): Total Length. Defines the entire packet size in bytes, including header and data.(2 bytes) 
ttl_list =         []                                                           # (int): Time to live (1 byte)
tos_list =         []                                                           # (int): Type of service. (1 byte)
id_list =          []                                                           # (int): Type of service. (1 byte)
proto_list =       []                                                           # (int): Protocol. This field defines the protocol used in the data portion of the IP datagram. (1 byte)
ip_chksum_list =   []                                                           # (int): Header checksum. (2 bytes)
frag_offset_list = []                                                           # (int): Fragment offset (13 bits)                                                     
df_flag_list =     []                                                           # (int): Don't fragment (1 bit)    
mf_flag_list =     []                                                           # (int): More fragments (1 bit)                                                                                   

# TCP header features
sport_list =       []                                                           # (int) : source port (2 bytes)
dport_list =       []                                                           # (int) : destination port (2 bytes)
seq_list =         []                                                           # sequence number
ack_list =         []                                                           # acknowledgement number
offset_list =      []                                                           # data offset in 32-bit words
window_list =      []                                                           # TCP window size
t_chksum_list =    []                                                           # (int): checksum (2 bytes)
urp_list =         []                                                           # urgent pointer
fin_flag_list =    []                                                           # (bool) : end of data
urg_flag_list =    []                                                           # (bool) : urgent pointer set
ack_flag_list =    []                                                           # (bool) : acknowledgement number set
syn_flag_list =    []                                                           # (bool) : synchronise sequence numbers
psh_flag_list =    []                                                           # (bool) : push
rst_flag_list =    []                                                           # (bool) : reset connection
ece_flag_list =    []                                                           # (bool) : ecn echo flag                                         
cwr_flag_list =    []                                                           # (bool) : congestion window reduced tcp flag
ns_flag_list =     []                                                           # (bool) : nonce sum flag                                                           

# UDP header features
ulen_list =        []                                                           # (int): Length. (2 bytes

# Label
label_list =       []
categorical_label_list = [] 

"""
   Definition of dictionary used to store all extracted packet features
"""

packet_features = {
    #Indexing and timestamps
    'packet_num': packet_num_list,
    'timestamp': timestamp_list,
    'datetime': datetime_list,
    
    #IP header features
    'version': version_list,
    'header_len' : header_len_list,
    'src_address': src_address_list,
    'dst_address': dst_address_list,
    'tot_len': tot_len_list,
    'ttl' : ttl_list,
    'tos': tos_list,
    'id': id_list,
    'proto': proto_list,
    'ip_chksum': ip_chksum_list,
    'frag_offset' : frag_offset_list,
    'df_flag': df_flag_list,
    'mf_flag': mf_flag_list,
    
    #TCP header features
    'sport': sport_list,
    'dport': dport_list,
    'offset': offset_list,
    'window': window_list,
    't_chksum': t_chksum_list,
    'urp': urp_list,
    'urg_flag': urg_flag_list,
    'fin_flag': fin_flag_list,
    'ack_flag': ack_flag_list,
    'syn_flag': syn_flag_list,
    'rst_flag': rst_flag_list,
    'psh_flag': psh_flag_list,
    'ece_flag': ece_flag_list,
    'cwr_flag': cwr_flag_list,
    'ns_flag' : ns_flag_list,
    
    #UDP header features 
    'ulen': ulen_list,
    
    #Label - Benign 0 and Attack 1
    'label': label_list,
    
    # Label
    'categorical_label' : categorical_label_list
}

# Open the PCAP file in binary mode
pcap_path = os.path.join(folder_path, file_name + '.pcap')
file = open(pcap_path, 'rb')
pcap = dpkt.pcapng.Reader(file)

# Count 
packets_processed = 0
packets_parsed = 0
non_ip_count = 0 
tcp_count = 0
udp_count = 0
 

for timestamp, buf in pcap:
    
    # Increment no of packets processed
    packets_processed +=1 
    
    if (packets_processed % 100000 == 0):
        print(packets_processed)
        print(len(packet_features['categorical_label']))
    
    # Check if we have processed the desired no of packets
    # (before continue statement as this may cause inf loop)
    #if (packets_processed == 700000):
     #   break
    
    # Before adding any entries to our dict, we want to make sure we don't include..
    # packets which don't use IP and TCP/UDP

    # Pass raw buffer through ethernet class
    eth = dpkt.ethernet.Ethernet(buf)   
    
    dpkt.ethernet.Ethernet.data
    
    # Access the data within the eth frame - which should be the IP packet
    ip = eth.data
    
    # Check if ethernet data contains an IP packet - continue if not
    if not isinstance(ip, dpkt.ip.IP):
       non_ip_count += 1
       continue
   
    # Access payload of IP packet - transport layer packet 
    transport_data = ip.data
        
    # Skip if not TCP or UDP
    if (not isinstance(transport_data, dpkt.tcp.TCP) and
        not isinstance(transport_data, dpkt.udp.UDP)):
        
            continue

    # Now we can start adding more entries to our dict

    # Binary labelling data - 0 for benign and 1 for attack
    ip_src = inet_to_str(ip.src)
    ip_dst = inet_to_str(ip.dst)
    
    packet_time = datetime.datetime.fromtimestamp(timestamp)                    # Convert timestamp to datetime
    packet_time -= datetime.timedelta(hours=4)                                  # Apply time shift to time in New Brunswick
    
    if weekday == 'Monday':                                                     # Monday has no attacks
        packet_features['label'].append(0)
        packet_features['categorical_label'].append('benign')
    else:
        found_match = False
        
        for attack in attacks:
            
            attack_ip_src, attack_ip_dst, start_time, end_time, label = attack
            
            check_time = start_time <= packet_time <= end_time
            check_match1 = (ip_src == attack_ip_src) and (ip_dst == attack_ip_dst)
            check_match2 = (ip_dst == attack_ip_src) and (ip_src == attack_ip_dst)
            check_match = check_match1 or check_match2
            
            if (attack_ip_src == vista):                                        # The one time vista attacks
                check_match = True                                              # It attacks everything so check match true
            
            if (check_time) and (check_match):
                
                packet_features['label'].append(1)
                packet_features['categorical_label'].append(label)
                found_match = True
                
                break
            
        if not found_match:
            packet_features['label'].append(0)
            packet_features['categorical_label'].append('benign')
                     
    # Index current packet
    packet_features['packet_num'].append(packets_processed)
    packet_features['timestamp'].append(timestamp)
    packet_features['datetime'].append(packet_time)

    # Extract IP header features
    packet_features['version'].append(ip.v)
    packet_features['header_len'].append(ip.hl)
    packet_features['src_address'].append(inet_to_str(ip.src))
    packet_features['dst_address'].append(inet_to_str(ip.dst))
    packet_features['tot_len'].append(ip.len)
    packet_features['ttl'].append(ip.ttl)
    packet_features['tos'].append(ip.tos)
    packet_features['id'].append(ip.id)
    packet_features['proto'].append(ip.p)
    packet_features['ip_chksum'].append(ip.sum)
    packet_features['frag_offset'].append(ip.offset)
    packet_features['df_flag'].append(ip.df)
    packet_features['mf_flag'].append(ip.mf)    
    
    # Extract TCP/UDP common header features
    packet_features['sport'].append(transport_data.sport)
    packet_features['dport'].append(transport_data.dport)
    packet_features['t_chksum'].append(transport_data.sum)
    
    # Check if TCP used in transport layer
    if isinstance(transport_data, dpkt.tcp.TCP):
        
        # Extract TCP specific features
        packet_features['offset'].append(transport_data.off)
        packet_features['window'].append(transport_data.win)
        packet_features['urp'].append(transport_data.urp)
        
        # Extracting tcp control flags using bitwise operations 
        urg_flag = bool(transport_data.flags & dpkt.tcp.TH_URG)
        fin_flag = bool(transport_data.flags & dpkt.tcp.TH_FIN)
        syn_flag = bool(transport_data.flags & dpkt.tcp.TH_SYN)
        ack_flag = bool(transport_data.flags & dpkt.tcp.TH_ACK)
        psh_flag = bool(transport_data.flags & dpkt.tcp.TH_PUSH)
        rst_flag = bool(transport_data.flags & dpkt.tcp.TH_RST)
        ece_flag = bool(transport_data.flags & dpkt.tcp.TH_ECE)
        cwr_flag = bool(transport_data.flags & dpkt.tcp.TH_CWR)
        ns_flag  = bool(transport_data.flags & dpkt.tcp.TH_NS)
        
        packet_features['urg_flag'].append(urg_flag)
        packet_features['fin_flag'].append(fin_flag)
        packet_features['syn_flag'].append(syn_flag)
        packet_features['ack_flag'].append(ack_flag)
        packet_features['psh_flag'].append(psh_flag)
        packet_features['rst_flag'].append(rst_flag)
        packet_features['ece_flag'].append(ece_flag)
        packet_features['cwr_flag'].append(cwr_flag)
        packet_features['ns_flag'].append(ns_flag)
        
        
        # Set UDP specific features to null
        packet_features['ulen'].append(False)
        
        tcp_count +=1
        
    # Check if UDP used in transport layer 
    if isinstance(transport_data, dpkt.udp.UDP):
        
        # Extract UDP specific features
        packet_features['ulen'].append(transport_data.ulen)
        
        # Set TCP specific features to false 
        packet_features['offset'].append(0)
        packet_features['window'].append(0)
        packet_features['urp'].append(0)
        packet_features['urg_flag'].append(False)
        packet_features['fin_flag'].append(False)
        packet_features['syn_flag'].append(False)
        packet_features['ack_flag'].append(False)
        packet_features['psh_flag'].append(False)
        packet_features['rst_flag'].append(False)
        packet_features['ece_flag'].append(False)
        packet_features['cwr_flag'].append(False)
        packet_features['ns_flag'].append(False)

        udp_count +=1  
    
    packets_parsed += 1
    

# Write the dict to csv
csv_path = os.path.join(folder_path, file_name + '.csv')

b2 = list(set(packet_features['categorical_label']))

write_to_csv(csv_path, packet_features)                           # Write to a csv      
    
b = list(set(packet_features['categorical_label']))



"""

Internet Protocol.

The Internet Protocol (IP) is the network layer communications protocol in the Internet protocol suite
for relaying datagrams across network boundaries. Its routing function enables internetworking, and
essentially establishes the Internet.

Attributes:
    __hdr__: Header fields of IP.
         _v_hl:
            v: (int): Version (4 bits) For IPv4, this is always equal to 4
            hl: (int): Internet Header Length (IHL) (4 bits)
        _flags_offset:
            rf: (int): Reserved bit (1 bit)
            df: (int): Don't fragment (1 bit)
            mf: (int): More fragments (1 bit)
            offset: (int): Fragment offset (13 bits)
        tos: (int): Type of service. (1 byte)
        len: (int): Total Length. Defines the entire packet size in bytes, including header and data.(2 bytes)
        id: (int): Identification. Uniquely identifying the group of fragments of a single IP datagram. (2 bytes)
        ttl: (int): Time to live (1 byte)
        p: (int): Protocol. This field defines the protocol used in the data portion of the IP datagram. (1 byte)
        sum: (int): Header checksum. (2 bytes)
        src: (int): Source address. This field is the IPv4 address of the sender of the packet. (4 bytes)
        dst: (int): Destination address. This field is the IPv4 address of the receiver of the packet. (4 bytes)

"""

"""

Transmission Control Protocol.

The Transmission Control Protocol (TCP) is one of the main protocols of the Internet protocol suite.
It originated in the initial network implementation in which it complemented the Internet Protocol (IP).

Attributes:
    sport - source port
    dport - destination port
    seq   - sequence number
    ack   - acknowledgement number
    off   - data offset in 32-bit words
    flags - TCP flags
    win   - TCP window size
    sum   - checksum
    urp   - urgent pointer
    opts  - TCP options buffer; call parse_opts() to parse
    
    
# TCP control flags
TH_FIN = 0x01  # end of data
TH_SYN = 0x02  # synchronize sequence numbers
TH_RST = 0x04  # reset connection
TH_PUSH = 0x08  # push
TH_ACK = 0x10  # acknowledgment number set
TH_URG = 0x20  # urgent pointer set


"""

"""
User Datagram Protocol.

User Datagram Protocol (UDP) is one of the core members of the Internet protocol suite.
With UDP, computer applications can send messages, in this case referred to as datagrams,
to other hosts on an Internet Protocol (IP) network. Prior communications are not required
in order to set up communication channels or data paths.

Attributes:
    __hdr__: Header fields of UDP.
        sport: (int): Source port. (2 bytes)
        dport: (int): Destination port. (2 bytes)
        ulen: (int): Length. (2 bytes)
        sum: (int): Checksum. (2 bytes)
        
"""