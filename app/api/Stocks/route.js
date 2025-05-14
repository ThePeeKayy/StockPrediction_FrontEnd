'use server'

import { addOrUpdateStock, removeOrUpdateStock } from "@/lib/actions"
import { getUserByEmail } from "@/lib/data"

export const POST = async(req) => {
    const {userEmail} = await req.json()
    const response = await getUserByEmail(userEmail);
    return new Response(JSON.stringify(response))
}

export const PATCH = async (req) => {
    const {userEmail, symbol, quantity} = await req.json();
    await addOrUpdateStock(userEmail, symbol, quantity);
    return new Response("Prompt updated successfully",{status:200});

}

export const DELETE = async (req) => {
    const {userEmail, symbol, quantity} = await req.json();
    await removeOrUpdateStock(userEmail, symbol, quantity)
    return new Response("Prompt updated successfully",{status:200});
}