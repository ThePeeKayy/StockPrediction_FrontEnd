'use server'

import { clearAndAddIndicators } from "@/lib/actions"
import { getIndicators } from "@/lib/data"

export const POST = async(req) => {
    const {date} = await req.json()
    const response = await getIndicators(date);
    return new Response(JSON.stringify(response))
}

export const PATCH = async (req) => {
    const {date, stockdata} = await req.json();
    await clearAndAddIndicators(date, stockdata);
    return new Response("Prompt updated successfully",{status:200});
}
