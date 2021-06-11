#! /usr/bin/env python3

import sys
l = len(sys.argv)

# Step 1. Getting the right file as mentioned in the arguments
if l!=2:
    print("Please provide valid netlist file name. Usage: "+sys.argv[0]+" <netlist-filename>")
    sys.exit()
else:
    filename = sys.argv[1]
try:
    with open(filename) as f:
        lines = f.readlines()
except FileNotFoundError:
    print("Please provide a valid netlist file name. File not found.")
    sys.exit()


# Step 2. Removing all comments and empty lines
for i,line in enumerate(lines):
    hashpos = line.find("#")
    if hashpos!=-1:
        lines[i] = line[0:hashpos]


for i,line in enumerate(lines):
    if line == '\n' or line == '':
        lines[i] = ''

# Step 3. Finding the circuit definition
all_lines = " ".join(lines)

strt_ckt_def = None
end_ckt_def = None

circuit = ".circuit"
endline = ".end"
resistor = 'R'
inductor = 'L'
capacitor = 'C'
voltage = 'V'
current = 'I'
VCVS = 'E'
VCCS = 'G'
CCVS = 'H'
CCCS = 'F'

if all_lines.find(circuit)==-1 or all_lines.find(endline)==-1:
    print("Please provide a valid netlist file. No circuit definition found.")
    sys.exit()

# Step 4. Handling multiple circuit definitions

index_circuit = [i for i in range(len(all_lines)) if all_lines.startswith(circuit,i)]
index_endline = [i for i in range(len(all_lines)) if all_lines.startswith(endline,i)]

if len(index_circuit)>1 or len(index_endline)>1:
    print("Please provide a valid netlist file. Multiple circuit definitions found.")
    sys.exit()

elif index_circuit[0] > index_endline[0]:
    print("Please provide a valid netlist file. "+endline+" occurs before "+circuit+".")

# Step 5. Separating the circuit definition from the rest of the file
for i,l in enumerate(lines):
    if l.startswith(circuit):
        strt_ckt_def = i
    elif l.startswith(endline):
        end_ckt_def = i

# Step 6. Splitting up each line into a set of tokens to work on
tokens = []

for l in lines[strt_ckt_def+1:end_ckt_def]:
    tokens.append(l.split())

# Step 7. Creating lists to store the values for each element

elem_names = []
from_nodes = []
to_nodes = []
control_low_nodes = []
control_high_nodes = []
control_currents = []
values = []

is_ok = False

# Step 8. Going through each valid line to add components to the right lists

for i,r in enumerate(tokens):
    line_number = i+strt_ckt_def+1
    
    # Step 8.1. Leaving out empty lines
    
    if not r:
        continue
        
    # Step 8.2. Handling R, L, C, V, I components
    
    elif r[0][0] in [resistor,inductor,capacitor,voltage,current]:
        try:
            float(r[3])
        except:
            print("Please enter a valid value, error on line "+str(line_number))
            sys.exit()
        if r[0] in elem_names:
            print("Please enter unique element names. Element \""+r[0]+"\" defined a second time on line "+str(line_number))
            sys.exit()
        elem_names.append(r[0])
        from_nodes.append(r[1])
        to_nodes.append(r[2])
        control_low_nodes.append(None)
        control_high_nodes.append(None)
        control_currents.append(None)
        values.append(r[3])
        
    # Step 8.3. Handling E and G components (voltage controlled)
    
    elif r[0][0] in [VCVS,VCCS]:
        is_ok = False
        if len(r) != 6:
            if len(r) > 6:
                if r[6].startswith('#'):
                    is_ok = True
            if not is_ok:
                print("Please enter valid SPICE commands, error on line "+str(line_number))
                sys.exit()
        else:
            try:
                float(r[5])
            except:
                print("Please enter a valid value, error on line "+str(line_number))
                sys.exit()
        elem_names.append(r[0])
        from_nodes.append(r[1])
        to_nodes.append(r[2])
        control_low_nodes.append(r[3])
        control_high_nodes.append(r[4])
        values.append(r[5])
        control_currents.append(None)
    
    # Step 8.3. Handling H and F components (current controlled)
    
    elif r[0][0] in [CCVS, CCCS]:
        is_ok = False
        if len(r) != 5:
            if len(r) > 5:
                if r[5].startswith('#'):
                    is_ok = True
            if not is_ok:
                print("Please enter a valid value, error on line "+str(line_number))
                sys.exit()
        else:
            try:
                float(r[4])
            except ValueError:
                print("Please enter a valid value, error on line "+str(line_number))
                sys.exit()
        elem_names.append(r[0])
        from_nodes.append(r[1])
        to_nodes.append(r[2])
        control_low_nodes.append(None)
        control_high_nodes.append(None)
        control_currents.append(r[3])
        values.append(r[4])
    
    # Step 8.4. Handling any non-circuit lines in between the circuit definition
    
    else:
        print("\"" + " ".join(r)+"\" - Unsupported component on line "+str(line_number))
        sys.exit()
    
# Step 9. Printing lines in reverse order with every token in opposite order of appearance

for i in reversed(range(len(elem_names))):
    if elem_names[i][0] in [resistor, inductor, capacitor, voltage, current]:
        print(" ".join([values[i],to_nodes[i],from_nodes[i],elem_names[i]]))
    elif elem_names[i][0] in [VCVS, VCCS]:
        print(" ".join([values[i],control_high_nodes[i],control_low_nodes[i],to_nodes[i],from_nodes[i],elem_names[i]]))
    elif elem_names[i][0] in [CCVS, CCCS]:
        print(" ".join([values[i],control_currents[i],to_nodes[i],from_nodes[i],elem_names[i]]))
