# 1. Base Image: Use a slim, modern Python
# This is a clean, minimal Linux OS with Python 3.12 pre-installed.
FROM python:3.12-slim

# 2. Set Environment Variables
# Tells Python to print logs immediately
ENV PYTHONUNBUFFERED=1

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Install Dependencies
# If change only python code will lead to faster build
COPY requirements.txt .

RUN python -m pip install --no-cache-dir -r requirements.txt

# 6. Set up Mount Points for Data and Logs
# This declares that these directories are intended for
# user-supplied data (volumes). This is documentation.
# The actual "linking" happens in the 'docker run' command.
VOLUME /app/sharded_data
VOLUME /app/logs

# 7. Expose the default port (documentation only)
# This just tells other developers that the container
# is *intended* to run on port 9000.
EXPOSE 9000

# 8. Define the runtime command
# This is the command that will run when the container starts.
CMD ["python", "services/agents/main_without_llm.py"]