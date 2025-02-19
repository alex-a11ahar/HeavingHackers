"""
EDIT: this function is incomplete/incorrect. fix it with the help of the
      comments provided
- function to perform random mutation
"""
def my_random(mut, inst_type, num_bits, bits_type):
    # Select bits to flip.
    m_index = cal_m_index(inst_type, num_bits, bits_type)

    # EDIT 1 START: Perform random mutation on the instruction.
    # `mut` is the binary string representing the instruction to mutate.
    # `m_index` contains the indices of the bits to flip in the string.

    # Convert the binary string `mut` to a list for easy manipulation.
    mut_list = list(mut)

    # Iterate through the indices in `m_index` and flip the corresponding bits.
    import random  # Ensure random module is available for randomization.
    for i in m_index:
        # Flip the bit randomly: '1' becomes '0' and '0' becomes '1'.
        mut_list[i] = '0' if mut_list[i] == '1' else '1'

    # Convert the mutated list back to a binary string.
    mut = ''.join(mut_list)

    # EDIT 1 END
    
    return mut
