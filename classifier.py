import sys

def classifier_function(param):
    # Your classifier logic here
    result = f"Processed {param}"
    return result

if __name__ == "__main__":
    input_param = sys.argv[1]  # Get command-line argument
    output = classifier_function(input_param)
    print(output)  # Print the result, which can be captured by the LLM script