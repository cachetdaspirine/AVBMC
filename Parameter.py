#  ____                  _
# / ___|   _   _   ___  | |_    ___   _ __ ___
# \___ \  | | | | / __| | __|  / _ \ | '_ ` _ \
#  ___) | | |_| | \__ \ | |_  |  __/ | | | | | |
# |____/   \__, | |___/  \__|  \___| |_| |_| |_|
#          |___/

Kmain=0.1
Kcoupling=0.868
Eps=0.01
KVOL=4.7
#----------------
J=0.0004
#----------------
SizeX=35
SizeY=35
NumberOfParticle=200

#  ____                                              _
# |  _ \    __ _   _ __    __ _   _ __ ___     ___  | |_    ___   _ __   ___
# | |_) |  / _` | | '__|  / _` | | '_ ` _ \   / _ \ | __|  / _ \ | '__| / __|
# |  __/  | (_| | | |    | (_| | | | | | | | |  __/ | |_  |  __/ | |    \__ \
# |_|      \__,_| |_|     \__,_| |_| |_| |_|  \___|  \__|  \___| |_|    |___/

TimeStepTot=1000
StatTime=TimeStepTot//100
BetaInitial=0
BetaFinal=1.6*10**2
Seed=25
DEG=0.0125
Pbias = 0.5
PInOut = 0.5
#  _____                           _                   _
# |_   _|   ___    _ __     ___   | |   ___     __ _  (_)   ___
#   | |    / _ \  | '_ \   / _ \  | |  / _ \   / _` | | |  / _ \
#   | |   | (_) | | |_) | | (_) | | | | (_) | | (_| | | | |  __/
#   |_|    \___/  | .__/   \___/  |_|  \___/   \__, | |_|  \___|
#                 |_|                          |___/
TopologieDown = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
TopologieUp = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
#TopologieDown = [(1,0),(-1,0),(0,1)]
#TopologieUp = [(1,0),(-1,0),(0,-1)]
