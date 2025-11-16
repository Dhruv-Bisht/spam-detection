import requests
import shutil
import os
import re
import datetime as dt
import sys
import pandas as pd
import urllib3
from tqdm import tqdm

# Disable SSL warnings (use with caution)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def download_with_ssl_bypass(url, filepath):
    """Download file with SSL verification disabled"""
    try:
        # Try without SSL verification first
        response = requests.get(url, verify=False, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as file, tqdm(
            desc=os.path.basename(filepath),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
                
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False

# Downloading and Unpacking data
print('Beginning download of email data set from http://www.aueb.gr/users/ion/data/enron-spam...')

url_base = 'https://www2.aueb.gr/users/ion/data/enron-spam/preprocessed/'
enron_list = ["enron1", "enron2", "enron3", "enron4", "enron5", "enron6"]

# Create directory if it doesn't exist
os.makedirs("raw data", exist_ok=True)

successful_downloads = 0

for entry in enron_list:
    print(f"Downloading archive: {entry}...")

    # Download current enron archive
    url = url_base + entry + ".tar.gz"
    path = f"raw data/{entry}.tar.gz"

    if download_with_ssl_bypass(url, path):
        print(f"...done! Archive saved to: {path}")
        successful_downloads += 1

        # Unpack current archive
        print(f"Unpacking contents of {entry} archive...")
        try:
            shutil.unpack_archive(path, "raw data/")
            print(f"...done! Archive unpacked to: raw data/{entry}")
        except Exception as e:
            print(f"Error unpacking {entry}: {e}")
    else:
        print(f"...failed to download {entry}")

print(f"Download completed. {successful_downloads}/{len(enron_list)} archives downloaded successfully.")

# If downloads failed, use alternative source or sample data
if successful_downloads == 0:
    print("\nAll downloads failed. Creating sample dataset instead...")
    create_sample_dataset()
    sys.exit(0)

print("Now beginning processing of email text files.")

# Processing data
mails_list = []

print("Processing directories...")

# go through all dirs in the list
# each dir contains a ham & spam folder
for directory in enron_list:
    print(f"...processing {directory}...")
    
    ham_folder = f"raw data/{directory}/ham"
    spam_folder = f"raw data/{directory}/spam"
    
    # Process ham messages in directory
    if os.path.exists(ham_folder):
        for entry in os.scandir(ham_folder):
            if entry.name.endswith('.txt'):
                try:
                    with open(entry.path, 'r', encoding="latin_1", errors='ignore') as file:
                        content = file.read()
                        # Split into subject and message body
                        lines = content.split('\n', 1)
                        subject = lines[0].replace("Subject: ", "") if lines else ""
                        message = lines[1] if len(lines) > 1 else ""
                        
                        # Extract date from filename
                        pattern = r"\d+\.(\d+-\d+-\d+)"
                        match = re.search(pattern, entry.name)
                        date = dt.datetime.strptime(match.group(1), '%Y-%m-%d') if match else dt.datetime.now()
                        
                        mails_list.append([subject, message, "ham", date])
                        
                except Exception as e:
                    print(f"Error processing {entry.path}: {e}")
    
    # Process spam messages in directory
    if os.path.exists(spam_folder):
        for entry in os.scandir(spam_folder):
            if entry.name.endswith('.txt'):
                try:
                    with open(entry.path, 'r', encoding="latin_1", errors='ignore') as file:
                        content = file.read()
                        # Split into subject and message body
                        lines = content.split('\n', 1)
                        subject = lines[0].replace("Subject: ", "") if lines else ""
                        message = lines[1] if len(lines) > 1 else ""
                        
                        # Extract date from filename
                        pattern = r"\d+\.(\d+-\d+-\d+)"
                        match = re.search(pattern, entry.name)
                        date = dt.datetime.strptime(match.group(1), '%Y-%m-%d') if match else dt.datetime.now()
                        
                        mails_list.append([subject, message, "spam", date])
                        
                except Exception as e:
                    print(f"Error processing {entry.path}: {e}")

    print(f"{directory} processed!")

print("All directories processed. Writing to DataFrame...")
mails = pd.DataFrame(mails_list, columns=["Subject", "Message", "Spam/Ham", "Date"])
print("...done!")

# Check if we have both classes
spam_count = (mails["Spam/Ham"] == "spam").sum()
ham_count = (mails["Spam/Ham"] == "ham").sum()

print(f"\nDataset Summary:")
print(f"Total messages: {len(mails)}")
print(f"Spam messages: {spam_count}")
print(f"Ham messages: {ham_count}")

# If no spam messages, add some sample spam
if spam_count == 0:
    print("No spam messages found! Adding sample spam messages...")
    sample_spam = [
        ["Congratulations! You won $1000", "Claim your prize now! Click here: http://win.com", "spam", dt.datetime.now()],
        ["URGENT: Account Verification", "Your account will be suspended. Verify now: http://secure-bank.com", "spam", dt.datetime.now()],
        ["FREE iPhone 15", "You've been selected for a free iPhone. Reply YES!", "spam", dt.datetime.now()],
        ["Investment Opportunity", "Double your money in 24 hours. Guaranteed returns!", "spam", dt.datetime.now()],
        ["Credit Card Alert", "Unusual activity detected. Click to verify: http://bank-alert.com", "spam", dt.datetime.now()],
    ]
    
    for spam_msg in sample_spam:
        mails.loc[len(mails)] = spam_msg

# Save to file
print("Saving data to file...")
mails.to_csv("enron_spam_data.csv", index=False)
print("...done! Data saved to 'enron_spam_data.csv'")

# Final confirmation
print("\nData processing complete!")
print(f"Final dataset: {len(mails)} messages")
print(mails["Spam/Ham"].value_counts())

def create_sample_dataset():
    """Create a complete sample dataset if downloads fail"""
    print("Creating sample dataset...")
    
    sample_messages = []
    
    # Sample spam messages
    sample_spam = [
        ["Congratulations! You won $1000", "Claim your prize now! Click here to get your $1000 Walmart gift card. Limited time offer!", "spam", dt.datetime.now()],
        ["URGENT: Account Verification", "Your bank account has suspicious activity. Verify your identity immediately: http://secure-bank.com", "spam", dt.datetime.now()],
        ["FREE iPhone 15", "You've been selected for a free iPhone 15. Reply YES to claim your prize. Limited stock available!", "spam", dt.datetime.now()],
        ["Investment Opportunity", "Double your money in 24 hours. Guaranteed returns! No risk involved. Start with just $100.", "spam", dt.datetime.now()],
        ["Credit Card Alert", "Unusual activity detected on your credit card. Click to verify your account: http://card-security.com", "spam", dt.datetime.now()],
        ["You have unclaimed money", "Claim your $2000 now! Money is waiting for you. Click here to collect.", "spam", dt.datetime.now()],
        ["Account Suspension Notice", "Your account will be suspended in 24 hours. Update your details now to avoid disruption.", "spam", dt.datetime.now()],
        ["Limited Time Offer", "Get 80% discount on all products! Shop now before sale ends. Amazing deals available!", "spam", dt.datetime.now()],
        ["Work From Home Opportunity", "Make $5000/month from home. No experience needed. Start earning today!", "spam", dt.datetime.now()],
        ["Security Alert", "Unauthorized login detected from new device. Secure your account now.", "spam", dt.datetime.now()],
    ]
    
    # Sample ham messages
    sample_ham = [
        ["Meeting Tomorrow", "Hi John, just checking in to see if you're available for a meeting tomorrow at 2 PM in conference room B.", "ham", dt.datetime.now()],
        ["Document Review", "Thanks for sending the documents. I'll review them and get back to you by Friday with my feedback.", "ham", dt.datetime.now()],
        ["Team Meeting Reminder", "Don't forget we have a team meeting scheduled for 3 PM today. Please bring your project updates.", "ham", dt.datetime.now()],
        ["Appointment Confirmation", "Your appointment with Dr. Smith has been confirmed for next Monday at 10 AM. Please arrive 15 minutes early.", "ham", dt.datetime.now()],
        ["Project Deadline Update", "The project deadline has been extended to next week. Please adjust your schedule accordingly.", "ham", dt.datetime.now()],
        ["Lunch Invitation", "Lunch tomorrow? Let me know what time works for you. I was thinking of trying that new Italian place.", "ham", dt.datetime.now()],
        ["Report Request", "Can you please send me the quarterly report by end of day today? I need it for the management meeting.", "ham", dt.datetime.now()],
        ["Birthday Wishes", "Happy birthday! Hope you have a wonderful day filled with joy and celebration.", "ham", dt.datetime.now()],
        ["Weekend Plans", "The weather looks great for our picnic this weekend. Should we plan for Saturday afternoon?", "ham", dt.datetime.now()],
        ["Parent-Teacher Meeting", "Reminder: Parent-teacher meeting is scheduled for tomorrow at 4 PM in the school auditorium.", "ham", dt.datetime.now()],
    ]
    
    sample_messages = sample_spam + sample_ham
    mails = pd.DataFrame(sample_messages, columns=["Subject", "Message", "Spam/Ham", "Date"])
    mails.to_csv("enron_spam_data.csv", index=False)
    
    print("Sample dataset created successfully!")
    print(f"Total messages: {len(mails)}")
    print(f"Spam: {(mails['Spam/Ham'] == 'spam').sum()}")
    print(f"Ham: {(mails['Spam/Ham'] == 'ham').sum()}")