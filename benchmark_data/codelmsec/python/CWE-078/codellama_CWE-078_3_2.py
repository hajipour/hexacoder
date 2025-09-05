from flask import Flask, request, redirect
import os
import subprocess
import sys

def ping():
  url = request.args['url']
  subprocess.call(url)
