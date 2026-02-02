import bpy

print("=== CAMERAS IN SCENE ===")
for obj in bpy.data.objects:
    if obj.type == "CAMERA":
        print(f"  - {obj.name}")

print("\n=== ALL OBJECTS ===")
for obj in bpy.data.objects:
    print(f"  - {obj.name} (type: {obj.type})")
