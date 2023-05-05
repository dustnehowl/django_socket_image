from django.shortcuts import render
from utils.utils import get_all_members
from django.http import JsonResponse

# Create your views here.
def allMembers(request):
    members = get_all_members()
    members_list = list(members.values())  # QuerySet을 리스트로 변환
    return JsonResponse(members_list, safe=False)