from constants import TEXT_COLUMN_NAME, VALUE_COLUMN_NAME

class FileParser: 
  def __init__(self): 
    pass

  # function to parse a file and return its data
  # @param path: string indicating the path to the file to parse
  # @param constantAdded: integer value to add to all sentiment values parsed from the file - used because certain loss functions do not allow negative values
  # @return: array of the data entries found in the file
  def parseFile(self, path, constantAdded = 0): 
    # lines = [] # entries of format {"text": text, "label": sentimentValue}
    datasetDict = {TEXT_COLUMN_NAME: [], VALUE_COLUMN_NAME: []}
    try: 
      with open(path, encoding="utf-8", errors="replace") as file:
        for x in file:
          indexOfEndOfText = x.rfind(",")
          # lines.append({"text": x[0: indexOfEndOfText], "label": x.replace("\n","")[indexOfEndOfText + 1:]})
          datasetDict[TEXT_COLUMN_NAME].append(x[0: indexOfEndOfText])
          datasetDict[VALUE_COLUMN_NAME].append(int(x.replace("\n","")[indexOfEndOfText + 1:]) + constantAdded)
        file.close()
    except FileNotFoundError:
      raise FileNotFoundError(f"File not found: {path}")
    except UnicodeDecodeError as e:
      raise ValueError(f"Cannot decode file {path} with utf-8 encoding: {e}")
    except PermissionError:
      raise PermissionError(f"Permission denied reading file: {path}")
    except Exception as e: 
      raise RuntimeError(f"Unexpected error reading file {path}: {e}")
    return datasetDict
    # return lines
