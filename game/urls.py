from django.urls import path

from . import views

urlpatterns = [
    path("", views.puzzle_list, name="puzzle_list"),
    path("puzzle/<int:pk>/", views.puzzle, name="puzzle"),
    path("puzzle/<int:pk>/check/", views.check_puzzle, name="check_puzzle"),
]
