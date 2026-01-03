def number_of_bits_required(number):
    bits_required = 0
    x = number
    while x > 0:
        x = x >> 1
        bits_required += 1
    return bits_required

def initialise_from_int(quantum_circuit, quantum_register, number):
    stored_so_far = 0
    i = 0
    while stored_so_far < number:
        this_bit = number & (1 << i)
        if this_bit > 0:
            #print("I will flip",i)
            quantum_circuit.x(quantum_register[i])
            stored_so_far += this_bit
            #print("That means I have stored", stored_so_far)
        i += 1
        if i > 10:
            raise ValueError


    
