import subprocess
import re
import math
import pandas as pd

'''
REMEMBER TO DELETE THE LOG EVERYTIME WHEN STARTING A NEW RUN 
'''

#gcc -O2 -fno-tree-vectorize stream.c -o stream_c.exe -DSTREAM_ARRAY_SIZE=20000000 -DSTRIDE=1000 -DSTREAM_TYPE=double 
#STREAM_ARRAY_SIZE=20000000 STRIDE=1000 cargo build --release 
c_make_command = "gcc -O2 stream.c -o stream_c.exe"
c_make_f32_command = "gcc -O2 stream.c -o stream_c_f32.exe"
c_make_f64_command = "gcc -O2 stream.c -o stream_c_f64.exe"
rust_make_command = " cargo build --release"
array_size_add_on = "STREAM_ARRAY_SIZE="
dash_D_add_on = " -D"
stride_add_on = "STRIDE="
stream_type_add_on = "STREAM_TYPE="
run_rust = "./target/release/sawtooth_stream"
run_c = "./stream_c.exe"
run_c_f32 = "./stream_c_f32.exe"
run_c_f64 = "./stream_c_f64.exe"
file_name = 'cache_bandwidth_log'
gcc_novec = '-fno-tree-vectorize'

hline = f"-------------------------------------------------------------"

def to_next_power_of_two(n):
   return 2**math.ceil(math.log(n, 2))

stride = 2000
counter = 8
diff = 1
stop = to_next_power_of_two(5592405) # 2^25=33554432, let's boost it to 2^27=134217728

# For cycle2 machine, L1 32KB, L2 should be 3MB, and L3 should be 30MB
# For Shaotong's machine, L1d is a 16 instances of 512KB, L2 is 16 instances of 16MB, and L3 is 2 instances of 128MB
# For Yifan's machine, L1d is 32KB, L2 is 32MB, and L3 is 2 instances of 768MB

fin_data = dict()

def run_make_command(command):
    try:
        # Run the make command in a subprocess
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Check if the command was successful (return code 0)
        if result.returncode == 0:
            return (result.stdout).strip()
        else:
            return f"Command failed with the following error: {result.stderr}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

def get_result(size, result, type):
   # Define regular expressions to match the data for Copy, Scale, Add, and Triad
   pattern = r"(CYCLIC|SAWTOOTH|RAND FORWARD FORWARD|RAND FORWARD BACKWARD|RAND BACKWARD BACKWARD):\s+(\d+\.?\d*e?[+-]?\d*|inf)\s+(\d+\.?\d*e?[+-]?\d*)\s+(\d+\.?\d*e?[+-]?\d*)\s+(\d+\.?\d*e?[+-]?\d*)\s+(\d+\.?\d*e?[+-]?\d*)\s+(\d+\.?\d*e?[+-]?\d*)"

   # Use the findall method to extract the data
   matches = re.findall(pattern, result)

   # Store the extracted data in a dictionary
   for match in matches:
      fin_data[match[0]] = (size, float(match[1]), float(match[2]), float(match[3]), float(match[4]), float(match[5]), float(match[6]))
   print(f'data: {fin_data}\n')

   df = pd.DataFrame.from_dict(fin_data, orient='index', columns = ['Size', 'Best Rate MB/s', 'Avg time', 'Min time', 'Max time', 'Access Times', 'Avg Time per Access'])
   print(df)
   df.to_csv(f'cache_bandwidth_log.csv', mode='a', header=False)

   # Print the extracted data
   # with open(file_name, 'a') as file:
   #    print(hline)
   #    file.write(f"{hline}\n")
   #    print(f"Array Size: {size} elements in {type}")
   #    file.write(f"Array Size: {size} elements in {type}\n")
   #    for function, values in data.items():
   #       print(f"{function}: Best Rate MB/s {values[0]}, Avg time {values[1]}, Min time {values[2]}, Max time {values[3]}, Access Times{values[4]}, Avg Time per Access{values[5]}")
   #       file.write(f"{function}: Best Rate MB/s {values[0]}, Avg time {values[1]}, Min time {values[2]}, Max time {values[3]}, Access Times{values[4]}, Avg Time per Access{values[5]}\n")
   #    print(f"{hline}\n")
   #    file.write(f"{hline}\n\n")
   #    file.flush()
   #    file.close()

if __name__ == "__main__":

   while counter <= stop:
      actual_size = to_next_power_of_two(counter)
      c_size = dash_D_add_on + array_size_add_on + str(actual_size)
      # c_stride = dash_D_add_on + stride_add_on + str(stride)
      # c_command = c_make_command + c_size + c_stride
      
      # rust_size = array_size_add_on + str(counter) + " "
      # rust_stride = stride_add_on + str(stride)
      # rust_command = rust_size + rust_stride + rust_make_command

      # f32 = dash_D_add_on + stream_type_add_on + "float"
      f64 = dash_D_add_on + stream_type_add_on + "double"

      # f32_command = c_make_f32_command + c_size + f32
      f64_command = c_make_f64_command + c_size + f64

      # run_make_command(f32_command)
      # print(f"f32 command: {f32_command}")
      # f32_result = run_make_command(run_c_f32)

      run_make_command(f64_command)
      # print(f"f64 command: {f64_command}")
      f64_result = run_make_command(run_c_f64)
      # get_result(f32_result, "float")
      get_result(actual_size, f64_result, "double")
      if actual_size > counter:
         counter = actual_size
      else:
         counter += diff
