"""
URL configuration for GrothendieckAi project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from qa.views import math_qa_view, history_view, home_view, library_view, delete_pdf, tutor_page_view, tutor_api_view
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views
from GrothendieckAi.views import register
from qa import views 
from django.contrib.auth.views import LogoutView


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home_view, name='home'),
    path('ask/', math_qa_view, name='ask'),
    path('library/', library_view, name='library'),
    path('history/', history_view, name='history'),
    path('login/', auth_views.LoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(next_page='home'), name='logout'),
    path('register/', register, name='register'), 
    path('history/edit/<int:pk>/', views.edit_question_history, name='edit_history'),
    path('history/delete/<int:pk>/', views.delete_question_history, name='delete_history'),
    path('pdf/delete/<int:pk>/', delete_pdf, name='delete_pdf'),
    path('tutor/', tutor_page_view, name='tutor'),       # For GET — render the page
    path('tutor/api', tutor_api_view, name='tutor_api'), 
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)