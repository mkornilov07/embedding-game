from django.contrib.auth.views import LoginView, LogoutView
from django.urls import path

from . import views

urlpatterns = [
    path("", views.puzzle_list, name="puzzle_list"),
    path("puzzle/<int:pk>/", views.puzzle, name="puzzle"),
    path("puzzle/<int:pk>/check/", views.check_puzzle, name="check_puzzle"),
    path("login/", LoginView.as_view(template_name="game/login.html"), name="login"),
    path("logout/", LogoutView.as_view(next_page="puzzle_list"), name="logout"),
    path("register/", views.register, name="register"),
    path("create/", views.create_puzzle, name="create_puzzle"),
    path("create/save/", views.save_puzzle, name="save_puzzle"),
    path("puzzle/<int:pk>/delete/", views.delete_puzzle, name="delete_puzzle"),
    path("generate/", views.generate_combinations, name="generate_combinations"),
]
