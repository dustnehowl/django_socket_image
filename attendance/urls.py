from django.urls import path

from . import views

urlpatterns = [
    path("members/all", views.allMembers, name="allMembers"),
]