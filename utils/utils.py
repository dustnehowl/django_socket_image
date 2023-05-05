from attendance.models import Member

def get_all_members():
    members = Member.objects.all()
    return members