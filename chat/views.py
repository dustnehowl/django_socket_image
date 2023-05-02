from django.shortcuts import render

# Create your views here.
# chat/views.py
from django.shortcuts import render


def index(request):
    return render(request, "chat/index.html")


def room(request, room_name):
    if room_name == "faceId":
        return render(request, "chat/faceId.html", {"room_name": room_name})
    else:
        return render(request, "chat/room.html", {"room_name": room_name})


