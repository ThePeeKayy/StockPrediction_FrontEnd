import mongoose from "mongoose"

const stockSchema = new mongoose.Schema({
    userId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        required: true
      },
      interestedStocks: {
        type: [String],
        required: false
      }
},
{timestamps:true}
)


const indicatorsSchema = new mongoose.Schema({
    date: {
      type: String,
      ref: 'dateUpdated',
      required: true
    },
    JSONdata: {
      type: [Object],
      required: true
    }
},
)

const itemSchema = new mongoose.Schema({
  symbol: {
    type: String,
    required: true
  },
  quantity: {
    type: Number,
    required: true
  }
}, { _id: false });

const userSchema = new mongoose.Schema({
    name: {
      type: String,
      required: true
    },
    id: {
        type: String,
        required: true
    },
    email: {
      type: String,
      required: true,
      unique: true
    },
    image: {
      type: String,
    },
    createdAt: {
      type: Date,
      default: Date.now
    },
    riskAppetite: {
        type: String,
        enum: ['low', 'medium', 'high'],
        default: 'medium'
      },
    stocks: [itemSchema]
  });

  

export const Item = mongoose.models.Item || mongoose.model('Item', itemSchema)
export const Stock = mongoose.models.Stock || mongoose.model('Stock',stockSchema);
export const User = mongoose.models.User || mongoose.model('User',userSchema)
export const Indicators = mongoose.models.Indicators || mongoose.model('Indicators',indicatorsSchema)