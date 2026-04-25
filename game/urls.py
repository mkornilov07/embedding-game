from django.contrib.auth.views import LoginView, LogoutView
from django.urls import path

from . import views

urlpatterns = [
    path("", views.puzzle_list, name="puzzle_list"),
    path("puzzle/<int:pk>/", views.puzzle, name="puzzle"),
    path("puzzle/<int:pk>/check/", views.check_puzzle, name="check_puzzle"),
    path("puzzle/<int:pk>/check-row/", views.check_row, name="check_row"),
    path("login/", LoginView.as_view(template_name="game/login.html"), name="login"),
    path("logout/", LogoutView.as_view(next_page="puzzle_list"), name="logout"),
    path("register/", views.register, name="register"),
    path("create/", views.create_puzzle, name="create_puzzle"),
    path("create/save/", views.save_puzzle, name="save_puzzle"),
    path("puzzle/<int:pk>/edit/", views.create_puzzle, name="edit_puzzle"),
    path("puzzle/<int:pk>/delete/", views.delete_puzzle, name="delete_puzzle"),
    path("puzzle/<int:pk>/pin/", views.pin_puzzle, name="pin_puzzle"),
    path("generate/", views.generate_combinations, name="generate_combinations"),
    path("duel/create/", views.create_duel, name="create_duel"),
    path("duel/<int:pk>/cancel/", views.cancel_duel, name="cancel_duel"),
    path("duel/<int:pk>/accept/", views.accept_duel, name="accept_duel"),
    path("duel/<int:pk>/decline/", views.decline_duel, name="decline_duel"),
    path("duel/<int:pk>/", views.duel_detail, name="duel_detail"),
    path("duel/<int:pk>/row-solved/", views.duel_row_solved, name="duel_row_solved"),
    path("duel/<int:pk>/surrender/", views.duel_surrender, name="duel_surrender"),
]
