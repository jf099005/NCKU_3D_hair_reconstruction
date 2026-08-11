import bpy

# Cycles
scene = bpy.context.scene
scene.render.engine = 'CYCLES'

# RTX 4070 / OptiX
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'OPTIX'
prefs.refresh_devices()

for device in prefs.devices:
    print(device.type, device.name)
    device.use = (device.type == 'OPTIX')

scene.cycles.device = 'GPU'

# Preview settings
scene.cycles.samples = 64

print("Cycles device:", scene.cycles.device)
print("Samples:", scene.cycles.samples)
