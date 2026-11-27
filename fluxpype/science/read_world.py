# A quick function to read output FLUX world data

# Import libraries
import numpy

from fluxpype import worldio


def rdworld(filename):

    # Some quick testing on storing this data in a class object
    class world(object):

        def __init__(self):
            self.fc = self.fc()
            self.fx = self.fx()

        # Flux concentrations
        class fc:
            def __init__(self):
                self.id = []
                self.x = []
                self.y = []
                self.z = []
                self.fl = []

        def add_fc(self, id, x, y, z, fl):
            self.fc.id.append(id)
            self.fc.x.append(x)
            self.fc.y.append(y)
            self.fc.z.append(z)
            self.fc.fl.append(fl)

        # Fluxons
        class fx:
            def __init__(self):
                self.id = []
                self.fc0 = []
                self.fc1 = []
                self.fl = []
                self.x = []
                self.y = []
                self.z = []

        def add_fx(self, id, fc0, fc1, fl, r0, r1, x, y, z):
            self.fx.id.append(id)
            self.fx.fc0.append(fc0)
            self.fx.fc1.append(fc1)
            self.fx.fl.append(fl)
            self.fx.x.append(numpy.array(([r0[0]] if r0 is not None else []) + x + ([r1[0]] if r1 is not None else [])))
            self.fx.y.append(numpy.array(([r0[1]] if r0 is not None else []) + y + ([r1[1]] if r1 is not None else [])))
            self.fx.z.append(numpy.array(([r0[2]] if r0 is not None else []) + z + ([r1[2]] if r1 is not None else [])))

    # Let there be light
    w = world()

    # Parse with the shared reader (both the input and the core form of the world)
    parsed = worldio.read(filename)
    for fc in parsed.concentrations:
        w.add_fc(fc[0], numpy.double(fc[1]), numpy.double(fc[2]), numpy.double(fc[3]), numpy.double(fc[4]))
    for ln in parsed.lines:
        r0 = ln.start
        r1 = ln.end
        w.add_fx(ln.id, ln.fc0, ln.fc1, numpy.double(ln.flux), r0, r1, ln.x, ln.y, ln.z)

    return w


# A quick comment on reading FLUX output world files:

# Flux concentrations
# NEW	100	0.314000	-0.059000	-0.948000	-1	(null)	0
# FC    ID      X               Y               Z               Flux    BND PT  Rad

# Line
# LINE	101	-302	-301	-1	100	1            7.61951 -2.51477 -7.52456   0.314 -0.059 -0.948
# Line  ID      Start   End     FC0     FC1     Flux        StX     StY     StZ     EndX    EndY    EndZ

# Vertex
# VERTEX	101	10650	1	7.203456	-2.377456	-7.113698
# Vertex        LineID  VtxID   Pos     X               Y               Z
