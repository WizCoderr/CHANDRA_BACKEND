from mongoengine import Document, StringField, DateTimeField
from datetime import datetime

class Image(Document):
    """MongoDB model for storing image metadata"""
    
    original_url = StringField(required=True)
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)
    
    meta = {
        'collection': 'images',
        'indexes': [
            {'fields': ['created_at']},
            {'fields': ['original_url'], 'unique': True}
        ]
    }
    
    def save(self, *args, **kwargs):
        if not self.created_at:
            self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        return super(Image, self).save(*args, **kwargs)
