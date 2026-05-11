
// use std::error::Error;
// use time::{
//     PrimitiveDateTime,
//     UtcOffset,
//     macros::format_description,
//     OffsetDateTime
// };

// pub fn to_offset_date_time(date_time: &str) -> Result<OffsetDateTime, Box<dyn Error>> {
//     // 1. Separate the datetime string and the timezone abbreviation
//     let (dt_part, tz_part) = date_time.rsplit_once(' ').unwrap();
//
//     // 2. Parse the datetime into a PrimitiveDateTime (no offset yet)
//     let format = format_description!("[year]-[month]-[day] [hour]:[minute]:[second]");
//     let primitive_dt = PrimitiveDateTime::parse(dt_part, &format)?;
//
//     // 3. Map the abbreviation to a UtcOffset
//     // For "EST" (Eastern Standard Time), the offset is UTC-5.
//     let offset = match tz_part {
//         "EST" => UtcOffset::from_hms(-5, 0, 0)?,
//         "EDT" => UtcOffset::from_hms(-4, 0, 0)?,
//         "UTC" => UtcOffset::UTC,
//         _ => panic!("Unknown timezone abbreviation"),
//     };
//
//     // 4. Combine them into an OffsetDateTime
//     let offset_dt = primitive_dt.assume_offset(offset);
//
//     Ok(offset_dt)
// }

use std::error::Error;
use chrono::{NaiveDateTime, TimeZone};
use chrono_tz::America::New_York;
use time::OffsetDateTime;

const FORMAT: &str = "%Y-%m-%d %H:%M:%S";

pub fn to_offset_date_time(datetime: &str) -> Result<OffsetDateTime, Box<dyn Error>> {
    // 1. The local date-time string (no offset info)
    let date_str = "2024-07-01 12:00:00";
    // let format = "%Y-%m-%d %H:%M:%S";

    // 2. Parse into a NaiveDateTime (no timezone yet)
    let naive_dt = NaiveDateTime::parse_from_str(date_str, FORMAT)?;

    // 3. Attach the specific Time Zone.
    // chrono-tz handles DST transitions automatically for the given date.
    let local_dt = New_York.from_local_datetime(&naive_dt).single()
        .ok_or("Ambiguous or invalid local time (DST transition)")?;

    // 4. Convert to UTC
    let utc_dt_chrono = local_dt.with_timezone(&chrono::Utc);

    // 5. Convert to time::OffsetDateTime
    // We use the Unix timestamp to ensure a perfect transfer.
    let final_utc = OffsetDateTime::from_unix_timestamp(utc_dt_chrono.timestamp())?;

    println!("Local String: {}", date_str);
    println!("UTC Result:   {}", final_utc);

    Ok(final_utc)
}



