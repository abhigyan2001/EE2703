#! /usr/bin/env python3
import sys
import cmath
import numpy as np

# start of class definitions
resistor = 'R'
inductor = 'L'
capacitor = 'C'
voltage = 'V'
current = 'I'
VCVS = 'E'
VCCS = 'G'
CCVS = 'H'
CCCS = 'F'

class Resistor:
    def __init__(self,name,from_node,to_node,value):
        self.name = name
        self.from_node = from_node
        self.to_node = to_node
        self.value = value
    def __str__(self):
        return str([self.name,self.from_node,self.to_node,self.value])
    def __repr__(self):
        return self.__str__()
        
class Inductor:
    def __init__(self,name,from_node,to_node,value):
        self.name = name
        self.from_node = from_node
        self.to_node = to_node
        self.value = value
    def __str__(self):
        return str([self.name,self.from_node,self.to_node,self.value])
    def __repr__(self):
        return self.__str__()

class Capacitor:
    def __init__(self,name,from_node,to_node,value):
        self.name = name
        self.from_node = from_node
        self.to_node = to_node
        self.value = value
    def __str__(self):
        return str([self.name,self.from_node,self.to_node,self.value])
    def __repr__(self):
        return self.__str__()
        
class V_Source:
    def __init__(self,name,acdc,from_node,to_node,value,phase=0):
        self.name = name
        self.from_node = from_node
        self.to_node = to_node
        self.value = value
        self.acdc = acdc
        if phase is not None:
            self.phase = phase
    def __str__(self):
        return str([self.name,self.from_node,self.to_node,self.value,self.acdc,self.phase])
    def __repr__(self):
        return self.__str__()
        
class I_Source:
    def __init__(self,name,acdc,from_node,to_node,value,phase):
        self.name = name
        self.from_node = from_node
        self.to_node = to_node
        self.value = value
        self.acdc = acdc
        self.phase = phase
    def __str__(self):
        return str([self.name,self.from_node,self.to_node,self.value,self.acdc,self.phase])
    def __repr__(self):
        return self.__str__()
        
class E:
    def __init__(self,name,acdc,from_node,to_node,ctrl_low,ctrl_high,value):
        self.name = name
        self.acdc = acdc
        self.from_node = from_node
        self.to_node = to_node
        self.ctrl_low = ctrl_low
        self.ctrl_high = ctrl_high
        self.value = value
    def __str__(self):
        return str([self.name,self.from_node,self.to_node,self.ctrl_low,self.ctrl_high,self.value,self.acdc])
    def __repr__(self):
        return self.__str__()
        
class G:
    def __init__(self,name,acdc,from_node,to_node,ctrl_low,ctrl_high,value):
        self.name = name
        self.acdc = acdc
        self.from_node = from_node
        self.to_node = to_node
        self.ctrl_low = ctrl_low
        self.ctrl_high = ctrl_high
        self.value = value
    def __str__(self):
        return str([self.name,self.from_node,self.to_node,self.ctrl_low,self.ctrl_high,self.value,self.acdc])
    def __repr__(self):
        return self.__str__()
        
class F:
    def __init__(self,name,acdc,from_node,to_node,ctrl_curr,value):
        self.name = name
        self.acdc = acdc
        self.from_node = from_node
        self.to_node = to_node
        self.ctrl_curr = ctrl_curr
        self.value = value
    def __str__(self):
        return str([self.name,self.from_node,self.to_node,self.ctrl_curr,self.value,self.acdc])
    def __repr__(self):
        return self.__str__()

class H:
    def __init__(self,name,acdc,from_node,to_node,ctrl_curr,value):
        self.name = name
        self.acdc = acdc
        self.from_node = from_node
        self.to_node = to_node
        self.ctrl_curr = ctrl_curr
        self.value = value
    def __str__(self):
        return str([self.name,self.from_node,self.to_node,self.ctrl_curr,self.value,self.acdc])
    def __repr__(self):
        return self.__str__()

def autochoose(nm,from_node,to_node,value,acdc="dc",ctrl_low=None,ctrl_high=None):
    if nm[0]==resistor:
        return Resistor(nm, from_node, to_node, value)
    elif nm[0]==inductor:
        return Inductor(nm, from_node, to_node, value)
    elif nm[0]==capacitor:
        return Capacitor(nm, from_node, to_node, value)
    elif nm[0]==voltage:
        return V_Source(nm, acdc, from_node, to_node, value, ctrl_low)
    elif nm[0]==current:
        return I_Source(nm, acdc, from_node, to_node, value, ctrl_low)
    elif nm[0]==VCVS:
        return E(nm, acdc, from_node, to_node, ctrl_low, ctrl_high, value)
    elif nm[0]==VCCS:
        return G(nm, acdc, from_node, to_node, ctrl_low, value)
    elif nm[0]==CCVS:
        return H(nm, acdc, from_node, to_node, ctrl_low, value)
    elif nm[0]==CCCS:
        return F(nm, acdc, from_node, to_node, ctrl_low, value)
    else:
        print("This component is not supported yet, please try a manual assignment")

# end of class definitions

pi = np.pi

# Step 1. Getting the right file as mentioned in the arguments
l = len(sys.argv)
if l!=2:
    print("Please provide valid netlist file name")
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
    sys.exit()

# Step 5. Separating the circuit definition from the rest of the file
for i,line in enumerate(lines):
    if line.startswith(circuit):
        strt_ckt_def = i
    elif line.startswith(endline):
        end_ckt_def = i

# Step 5.1. Finding AC or DC

cirtyp = 'dc' #setting to dc by default
freq = 0      #setting frequency to 0 by default
for line in lines[end_ckt_def:]:
    if '.ac' in line:
        tk = line.split()
        cirtyp = 'ac'
        try:
            freq = float(tk[2])
        except:
            print("Please enter a valid frequency.")
            sys.exit()

# Step 6. Splitting up each line into a set of tokens to work on
tokens = []

for l in lines[strt_ckt_def+1:end_ckt_def]:
    tokens.append(l.split())

# Step 7. Creating lists to store nodes, components and circuit type

components = []

all_nodes = []
elem_names = []
is_ok = False

# Step 8. Going through each valid line to add components to the right lists
# Reading in the circuit components

for i,r in enumerate(tokens):
    line_number = i+strt_ckt_def+2
    
    # Step 8.0. Leaving out empty lines
    
    if not r:
        continue
    
    # Step 8.2. Handling R, L, C components
    
    elif r[0][0] in [resistor,inductor,capacitor]:
        try:
            val = float(r[3])
        except:
            print("Please enter a valid value, error on line "+str(line_number))
            sys.exit()
        if r[0] in elem_names:
            print("Please enter unique element names. Element \""+r[0]+"\" defined a second time on line "+str(line_number))
            sys.exit()
        all_nodes.append(r[1])
        all_nodes.append(r[2])
        
        if r[0][0] == capacitor:
            if freq!=0:
                val = 1/(2*pi*1j*freq*val)
            else:
                val = float('inf')
        elif r[0][0] == inductor:
            val = 2*pi*1j*freq*val
        
        components.append(autochoose(r[0],r[1],r[2],val))
    
    # Step 8.3. Handling V and I components
    
    elif r[0][0] in [voltage,current]:
        try:
            float(r[4])
            if cirtyp == 'ac':
                float(r[5])
        except:
            print("Please enter a valid value, error on line "+str(line_number))
            sys.exit()
        if r[0] in elem_names:
            print("Please enter unique element names. Element \""+r[0]+"\" defined a second time on line "+str(line_number))
            sys.exit()
        all_nodes.append(r[1])
        all_nodes.append(r[2])
        if len(r) == 6:
            components.append(autochoose(r[0],r[1],r[2],float(r[4])/2,r[3],r[5]))
        else:
            components.append(autochoose(r[0],r[1],r[2],r[4],r[3]))
    
    # Step 8.4. Handling E and G components (voltage controlled)
    
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
        all_nodes.append(r[1])
        all_nodes.append(r[2])
        all_nodes.append(r[3])
        all_nodes.append(r[4])
        
        components.append(autochoose(r[0],r[1],r[2],r[5],cirtyp,r[3],r[4]))
    
    # Step 8.5. Handling H and F components (current controlled)
    
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
        all_nodes.append(r[1])
        all_nodes.append(r[2])
        
        components.append(autochoose(r[0],r[1],r[2],r[4],cirtyp,r[3]))
    # Step 8.6. Handling any non-circuit lines in between the circuit definition
    
    else:
        print("\"" + " ".join(r)+"\" - Unsupported component on line "+str(line_number))
        sys.exit()
    elem_names.append(components[-1].name) #used to keep track of duplicate components

# Step 9. Filling up the Matrix

# Step 9.1 Finding the number of Nodes as number of unique entries 
# in the combined set of from_nodes, to_nodes, control_low_nodes and control_high_nodes

unique_nodes = list(set(all_nodes))
if 'GND' in unique_nodes:
    unique_nodes.remove("GND")
num_nodes = len(unique_nodes)

unique_nodes.sort()

# Step 9.2 Finding the number of Voltage Sources
num_v = 0
for i in components:
    if isinstance(i,V_Source):
        num_v += 1

# Step 9.3 Forming an empty Numpy Matrix of dimensions (num_v + num_nodes)*(num_v + num_nodes)

dim = num_nodes+num_v
A = np.zeros([dim,dim],dtype = complex)
B = np.zeros([dim],dtype = complex)
X = np.zeros([dim],dtype = complex)

# Step 9.4 Inputting all component equations

# Step 9.4.1 Inputting Resistors, Capacitors and Inductors
for comp in components:
    if isinstance(comp,Resistor) or isinstance(comp,Capacitor) or isinstance(comp,Inductor):
        if comp.from_node!='GND':
            try:
                A[unique_nodes.index(comp.from_node)][unique_nodes.index(comp.from_node)]+= 1/complex(comp.value)
                if comp.to_node!='GND':
                    A[unique_nodes.index(comp.from_node)][unique_nodes.index(comp.to_node)]-= 1/complex(comp.value)
            except:
                pass
        if comp.to_node!='GND':
            try:
                if comp.from_node!='GND':
                    A[unique_nodes.index(comp.to_node)][unique_nodes.index(comp.from_node)]-= 1/complex(comp.value)
                A[unique_nodes.index(comp.to_node)][unique_nodes.index(comp.to_node)]+= 1/complex(comp.value)
            except:
                print("Error")
                sys.exit()

# Step 9.4.1 Inputting Voltage and Current Sources
for comp in components:
    if isinstance(comp,V_Source):
        num=0
        if comp.from_node!='GND':
            A[num_nodes+num][unique_nodes.index(comp.from_node)]-=1
            A[unique_nodes.index(comp.from_node)][num_nodes+num]+=1
        if comp.to_node!='GND':
            A[num_nodes+num][unique_nodes.index(comp.to_node)]+=1
            A[unique_nodes.index(comp.to_node)][num_nodes+num]-=1
        B[num_nodes+num]+=complex(comp.value)
        num+=1
for comp in components:    
    if isinstance(comp,I_Source):
        for comp in components[4]:
            if comp.from_node!='GND':
                B[unique_nodes.index(comp.from_node)]-=complex(comp.value)
            if comp.to_node!='GND':
                B[unique_nodes.index(comp.to_node)]+=complex(comp.value)
# Step 9.5 Solving the matrix equation
#print(A)
#print(B)
try:
    X=np.linalg.solve(A,B)
except np.linalg.linalg.LinAlgError:
    print("Singular matrix")
    sys.exit()

#print(X)

#printing results
if "dc"==cirtyp:
    print(' ')
if "ac"==cirtyp:
    print(' ')
    print("Voltages and currents (rms) are in polar form (r,\u03B8)")
    print(' ')


for i in range(num_nodes):
    print("Voltage at node", end=' ')
    print(unique_nodes[i],end=' ')
    print("is", end=' ')
    if "dc"==cirtyp:
        if X[i].real < 1e-15:
            print(0)
        else:
            print(X[i].real)
    if "ac"==cirtyp:
        if X[i].real < 1e-15:
            X[i]=complex(0,0)
        print(cmath.polar(X[i]))

for i,comp in enumerate(components):
    if isinstance(comp,V_Source):
        print("Current through voltage source", end=' ')
        print(comp.name,end=' ')
        print("is", end=' ')
        if "dc"==cirtyp:
            if X[i+num_nodes].real < 1e-15:
                print(0)
            else:
                print(X[i+num_nodes].real)
        if "ac"==cirtyp:
            if abs(X[i+num_nodes]) < 1e-15:
                X[i+num_nodes]=complex(0,0)
            print(cmath.polar(X[i+num_nodes]))