'use server'
import { revalidatePath } from "next/cache";
import { Stock, User, Indicators } from "./models";
import { connectToDB } from "./utils";
import { redirect } from "next/navigation";

export const clearAndAddIndicators = async (date, newItemData) => {
  try {
    await connectToDB();

    await Indicators.deleteMany({});

    const newItem = new Indicators({ date, JSONdata: newItemData });
    await newItem.save();
    
    
  } catch (error) {
    console.error('Error clearing and adding item:', error);
    throw error;
  }
  revalidatePath('/');
  redirect('/');
};

export async function addOrUpdateStock(userEmail, stockSymbol, quantity) {
  try {
    await connectToDB();
    const user = await User.findOne({ email: userEmail });
    if (!user) {
      throw new Error(`User not found ${userEmail}`);
    }

    const stockIndex = user?.stocks?.findIndex(stock => stock.symbol === stockSymbol);

    if (stockIndex > -1) {
      user.stocks[stockIndex].quantity += Number(quantity);
    } else {
      user.stocks.push({ symbol: stockSymbol, quantity: Number(quantity) });
    }

    await user.save();
    return user;
  } catch (error) {
    console.error('Error updating stock:', error);
    throw error;
  }
}

export async function removeOrUpdateStock(userEmail, stockSymbol, quantity) {
  
  try {
    await connectToDB();
    const user = await User.findOne({ email: userEmail });

    if (!user) {
      throw new Error('User not found');
    }
    const stockIndex = user.stocks.findIndex(stock => stock.symbol === stockSymbol);

    if (stockIndex > -1) {
      if (user.stocks[stockIndex].quantity > quantity) {
        user.stocks[stockIndex].quantity -= quantity
      } else {
        user.stocks = user.stocks.filter(stock => stock.symbol !== stockSymbol)
      };
    } 

    await user.save();
    return user;
  } catch (error) {
    console.error('Error removing stock:', error);
    throw error;
  }
}

export const addStockInterest = async (userId, symbolName) => {
      await connectToDB();
      const userStocks = await Stock.findOne({ userId });
      if (userStocks) {
        await UserInterestedStocks.updateOne(
          { userId },
          { $addToSet: { interestedStocks: symbolName } } // Use $addToSet to avoid duplicates
        );
      } else {
        const newStock = new UserInterestedStocks({
          userId,
          interestedStocks: [symbolName]
        });
        await newStock.save();
      }
    revalidatePath('/');
  };
  
export const deleteStockInterest = async (userId, symbolName) => {
try {
    await connectToDB();
  
    await Stock.updateOne(
    { userId },
    { $pull: { interestedStocks: symbolName } }
    );
  
    } catch (error) {
        console.log('Error removing stock:', error);
    }
revalidatePath('/');
};
  
export const addUser = async(formData)=>{
    const {name, email, image, risk} = formData;
    try {
        await connectToDB()
        const date = new Date()
        let id = Math.random().toString(36).substring(2, 6)
        const newUser = new User({
            name, id, email, image, date, risk
        })

        await newUser.save()
    } catch (error) {
        
    }
    
    revalidatePath('/')
    redirect('/')
}