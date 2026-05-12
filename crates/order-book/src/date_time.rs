
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
use chrono_tz::{
    America::New_York,
    America::Chicago,
    America::Denver,
    America::Los_Angeles,
    UTC,
};
use databento::dbn::StatUpdateAction::New;
use time::OffsetDateTime;

const FORMAT: &str = "%Y-%m-%d %H:%M:%S";

pub fn to_offset_date_time(date_time_str: &str) -> Result<OffsetDateTime, Box<dyn Error>> {
    let (dt_part, tz_part) = date_time_str.rsplit_once(' ').unwrap();
    let naive_dt = NaiveDateTime::parse_from_str(dt_part, FORMAT)?;

    let local_dt = match tz_part {
        "ET" => New_York.from_local_datetime(&naive_dt),                // Eastern
        "CT" => Chicago.from_local_datetime(&naive_dt),                 // Central
        "MT" => Denver.from_local_datetime(&naive_dt),                  // Mountain
        "PT" => Los_Angeles.from_local_datetime(&naive_dt),             // Pacific
        "UTC" => UTC.from_local_datetime(&naive_dt),                    // UTC
        _ => return Err(format!("Invalid timezone: {}", tz_part).into()),
    };

    let local_dt = local_dt.single()
        .ok_or("Ambiguous or invalid local time (DST transition)")?;
    let utc_dt_chrono = local_dt.with_timezone(&chrono::Utc);
    let final_utc = OffsetDateTime::from_unix_timestamp(utc_dt_chrono.timestamp())?;

    Ok(final_utc)
}



