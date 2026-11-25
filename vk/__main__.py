import sys, importlib
CMDS = {"paper": "vk.paper", "cites": "vk.cites", "build": "vk.build", "qa": "vk.qa", "say": "vk.tts", "new": "vk.build", "fetch": "vk.walk"}
if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
    sys.exit("usage: ./vk {paper|cites|build|qa|say|new|fetch} ...   (add -h to any command)")
cmd = sys.argv[1]
mod = importlib.import_module(CMDS[cmd])
if cmd == "fetch":
    mod.fetch_main(sys.argv[2:])
else:
    mod.main(sys.argv[2:] if cmd != "new" else ["new", *sys.argv[2:]])
