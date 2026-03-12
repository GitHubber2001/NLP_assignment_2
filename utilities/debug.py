import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store", default="true")
args = parser.parse_args()

if args.debug.lower() == "true":
    DEBUG_ENABLED = True
elif args.debug.lower() == "false":
    DEBUG_ENABLED = False
else:
    raise RuntimeError("Invalid argument given for --debug flag")
