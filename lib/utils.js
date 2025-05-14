'use server';
import mongoose from 'mongoose';

// Persistent connection object to track connection status
let connection = {};

export const connectToDB = async () => {
  // Check if already connected to avoid multiple connections
  if (connection.isConnected) {
    console.log('Already connected to MongoDB');
    return;
  }

  try {
    const db = await mongoose.connect(process.env.MONGO, {
      useNewUrlParser: true,
      useUnifiedTopology: true,
      serverSelectionTimeoutMS: 15000,
    });
    await mongoose.connection.db.admin().command({ ping: 1 })

    connection.isConnected = true;
  } catch (error) {
    console.error('Error connecting to MongoDB:', error);
    throw new Error('MongoDB connection failed');
  }
};