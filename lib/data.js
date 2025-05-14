'use server'
import { Stock, User, Indicators } from "./models"
import { connectToDB } from "./utils"

export const getIndicators = async (date) => {
  try {
    await connectToDB();

    const indicators = await Indicators.findOne({ date });
    if (!indicators) {
      return false
    }
    return indicators;
  } catch (error) {
    console.log('Error fetching user:', error);
    throw error; // Re-throw the error for further handling if needed
  }
};

export const getUserByEmail = async (email) => {
    try {
      await connectToDB();
  
      const user = await User.findOne({ email });
      if (!user) {
        return false
      }
  
      return user;
    } catch (error) {
      console.log('Error fetching user:', error);
      throw error; // Re-throw the error for further handling if needed
    }
  };

  export const getUserStocks = async(userId) => {
    try {
        await connectToDB();
        const userPreference = await Stock.findOne({ userId });
        if (!userPreference) {
            return []
        }

        return userPreference
    } catch (error) {
        console.log('Error fetching stocks:', error);
    }
  }