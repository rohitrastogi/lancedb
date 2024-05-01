// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

pub use arrow_array;
use std::{mem, pin::Pin, sync::Arc};

pub use arrow_schema;
use futures::{Stream, StreamExt};

#[cfg(feature = "polars")]
use {
    arrow::ffi::from_ffi_and_data_type,
    polars::datatypes,
    polars::frame::ArrowChunk,
    polars::prelude::{DataFrame, Schema, Series},
};

use crate::error::Result;

/// An iterator of batches that also has a schema
pub trait RecordBatchReader: Iterator<Item = Result<arrow_array::RecordBatch>> {
    /// Returns the schema of this `RecordBatchReader`.
    ///
    /// Implementation of this trait should guarantee that all `RecordBatch`'s returned by this
    /// reader should have the same schema as returned from this method.
    fn schema(&self) -> Arc<arrow_schema::Schema>;
}

/// A simple RecordBatchReader formed from the two parts (iterator + schema)
pub struct SimpleRecordBatchReader<I: Iterator<Item = Result<arrow_array::RecordBatch>>> {
    pub schema: Arc<arrow_schema::Schema>,
    pub batches: I,
}

impl<I: Iterator<Item = Result<arrow_array::RecordBatch>>> Iterator for SimpleRecordBatchReader<I> {
    type Item = Result<arrow_array::RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        self.batches.next()
    }
}

impl<I: Iterator<Item = Result<arrow_array::RecordBatch>>> RecordBatchReader
    for SimpleRecordBatchReader<I>
{
    fn schema(&self) -> Arc<arrow_schema::Schema> {
        self.schema.clone()
    }
}

/// A stream of batches that also has a schema
pub trait RecordBatchStream: Stream<Item = Result<arrow_array::RecordBatch>> {
    /// Returns the schema of this `RecordBatchStream`.
    ///
    /// Implementation of this trait should guarantee that all `RecordBatch`'s returned by this
    /// stream should have the same schema as returned from this method.
    fn schema(&self) -> Arc<arrow_schema::Schema>;
}

/// A boxed RecordBatchStream that is also Send
pub type SendableRecordBatchStream = Pin<Box<dyn RecordBatchStream + Send>>;

impl<I: lance::io::RecordBatchStream + 'static> From<I> for SendableRecordBatchStream {
    fn from(stream: I) -> Self {
        let schema = stream.schema();
        let mapped_stream = Box::pin(stream.map(|r| r.map_err(Into::into)));
        Box::pin(SimpleRecordBatchStream {
            schema,
            stream: mapped_stream,
        })
    }
}

/// A simple RecordBatchStream formed from the two parts (stream + schema)
#[pin_project::pin_project]
pub struct SimpleRecordBatchStream<S: Stream<Item = Result<arrow_array::RecordBatch>>> {
    pub schema: Arc<arrow_schema::Schema>,
    #[pin]
    pub stream: S,
}

impl<S: Stream<Item = Result<arrow_array::RecordBatch>>> Stream for SimpleRecordBatchStream<S> {
    type Item = Result<arrow_array::RecordBatch>;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let this = self.project();
        this.stream.poll_next(cx)
    }
}

impl<S: Stream<Item = Result<arrow_array::RecordBatch>>> RecordBatchStream
    for SimpleRecordBatchStream<S>
{
    fn schema(&self) -> Arc<arrow_schema::Schema> {
        self.schema.clone()
    }
}

/// A trait for converting incoming data to Arrow
///
/// Integrations should implement this trait to allow data to be
/// imported directly from the integration.  For example, implementing
/// this trait for `Vec<Vec<...>>` would allow the `Vec` to be directly
/// used in methods like [`crate::connection::Connection::create_table`]
/// or [`crate::table::Table::add`]
pub trait IntoArrow {
    /// Convert the data into an Arrow array
    fn into_arrow(self) -> Result<Box<dyn arrow_array::RecordBatchReader + Send>>;
}

impl<T: arrow_array::RecordBatchReader + Send + 'static> IntoArrow for T {
    fn into_arrow(self) -> Result<Box<dyn arrow_array::RecordBatchReader + Send>> {
        Ok(Box::new(self))
    }
}

/// When interpreting Polars dataframes as polars-arrow record batches,
/// whether to use Arrow string/binary view types instead of the standard
/// Arrow string/binary types.
/// For now, we will not use string view types because conversions
/// for string view types from polars-arrow to arrow-rs are not yet implemented.
/// See: https://lists.apache.org/thread/w88tpz76ox8h3rxkjl4so6rg3f1rv7wt for the
/// differences in the types.
#[cfg(feature = "polars")]
const POLARS_ARROW_FLAVOR: bool = false;

#[cfg(feature = "polars")]
/// An iterator of record batches formed from a Polars DataFrame.
pub struct PolarsDataFrameRecordBatchReader {
    chunks: std::vec::IntoIter<ArrowChunk>,
    arrow_schema: Arc<arrow_schema::Schema>,
}

#[cfg(feature = "polars")]
impl PolarsDataFrameRecordBatchReader {
    /// Creates a new `PolarsDataFrameRecordBatchReader` from a given Polars DataFrame.
    /// If the input dataframe does not have aligned chunks, this function undergoes
    /// the costly operation of reallocating each series as a single contigous chunk.
    pub fn new(mut df: DataFrame) -> Self {
        df.align_chunks();
        let arrow_fields: Vec<arrow_schema::Field> = df
            .schema()
            .into_iter()
            .map(|(name, df_dtype)| {
                let polars_arrow_dtype = df_dtype.to_arrow(POLARS_ARROW_FLAVOR);
                let polars_field =
                    polars_arrow::datatypes::Field::new(name.clone(), polars_arrow_dtype, true);
                let polars_c_schema = polars_arrow::ffi::export_field_to_c(&polars_field);
                let arrow_c_schema: arrow::ffi::FFI_ArrowSchema =
                    unsafe { mem::transmute(polars_c_schema) };
                let arrow_rs_dtype = arrow_schema::DataType::try_from(&arrow_c_schema).unwrap();
                arrow_schema::Field::new(name, arrow_rs_dtype, true)
            })
            .collect();
        Self {
            chunks: df
                .iter_chunks(POLARS_ARROW_FLAVOR)
                .collect::<Vec<ArrowChunk>>()
                .into_iter(),
            arrow_schema: Arc::new(arrow_schema::Schema::new(arrow_fields)),
        }
    }
}

#[cfg(feature = "polars")]
impl Iterator for PolarsDataFrameRecordBatchReader {
    type Item = std::result::Result<arrow_array::RecordBatch, arrow_schema::ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.chunks.next().map(|chunk| {
            let columns: Vec<arrow_array::ArrayRef> = chunk
                .into_arrays()
                .into_iter()
                .zip(self.arrow_schema.clone().fields.into_iter())
                .map(|(polars_array, arrow_field)| {
                    convert_polars_array_to_arrow_rs_array(
                        polars_array,
                        arrow_field.data_type().to_owned(),
                    )
                })
                .collect();
            arrow_array::RecordBatch::try_new(self.arrow_schema.clone(), columns)
        })
    }
}

#[cfg(feature = "polars")]
fn convert_polars_array_to_arrow_rs_array(
    polars_array: Box<dyn polars_arrow::array::Array>,
    arrow_datatype: arrow_schema::DataType,
) -> arrow_array::ArrayRef {
    let polars_c_array = polars_arrow::ffi::export_array_to_c(polars_array);
    let arrow_c_array = unsafe { mem::transmute(polars_c_array) };
    arrow_array::make_array(
        unsafe { from_ffi_and_data_type(arrow_c_array, arrow_datatype) }.unwrap(),
    )
}

#[cfg(feature = "polars")]
impl arrow_array::RecordBatchReader for PolarsDataFrameRecordBatchReader {
    fn schema(&self) -> Arc<arrow_schema::Schema> {
        self.arrow_schema.clone()
    }
}

/// A trait for converting the result of a LanceDB query into a Polars DataFrame with aligned
/// chunks. The resulting Polars DataFrame will have aligned chunks, but the series's
/// chunks are not guaranteed to be contiguous.
#[cfg(feature = "polars")]
pub trait IntoPolars {
    fn into_polars(self) -> impl std::future::Future<Output = Result<DataFrame>> + Send;
}

#[cfg(feature = "polars")]
impl IntoPolars for SendableRecordBatchStream {
    async fn into_polars(mut self) -> Result<DataFrame> {
        let arrow_schema = self.schema();
        let polars_schema = convert_arrow_schema_to_polars_df_schema(&arrow_schema);
        let mut acc_df: DataFrame = DataFrame::from(&polars_schema);
        while let Some(record_batch) = self.next().await {
            let new_df = convert_record_batch_to_polars_df(&record_batch?, &polars_schema)?;
            acc_df = acc_df.vstack(&new_df)?;
        }
        Ok(acc_df)
    }
}

#[cfg(feature = "polars")]
fn convert_arrow_schema_to_polars_df_schema(
    arrow_schema: &arrow_schema::Schema,
) -> polars::prelude::Schema {
    use polars_arrow::ffi::import_field_from_c;

    polars::prelude::Schema::from_iter(arrow_schema.fields().iter().map(|arrow_field| {
        let arrow_rs_dtype = arrow_field.data_type();
        let arrow_c_schema = arrow::ffi::FFI_ArrowSchema::try_from(arrow_rs_dtype).unwrap();
        let polars_c_schema: polars_arrow::ffi::ArrowSchema =
            unsafe { mem::transmute(arrow_c_schema) };
        let polars_arrow_field = unsafe { import_field_from_c(&polars_c_schema) }.unwrap();
        polars::prelude::Field::new(
            arrow_field.name(),
            datatypes::DataType::from(polars_arrow_field.data_type()),
        )
    }))
}

#[cfg(feature = "polars")]
fn convert_record_batch_to_polars_df(
    record_batch: &arrow::record_batch::RecordBatch,
    polars_schema: &Schema,
) -> Result<DataFrame> {
    let mut columns: Vec<Series> = Vec::with_capacity(record_batch.num_columns());

    for (i, column) in record_batch.columns().iter().enumerate() {
        let polars_df_dtype = polars_schema.try_get_at_index(i)?.1;
        let polars_arrow_dtype = polars_df_dtype.to_arrow(POLARS_ARROW_FLAVOR);
        let polars_array = convert_arrow_rs_array_to_polars_array(column, polars_arrow_dtype);
        columns.push(Series::from_arrow(
            polars_schema.try_get_at_index(i)?.0,
            polars_array,
        )?);
    }

    Ok(DataFrame::from_iter(columns))
}

#[cfg(feature = "polars")]
fn convert_arrow_rs_array_to_polars_array(
    arrow_rs_array: &Arc<dyn arrow_array::Array>,
    polars_arrow_dtype: polars::datatypes::ArrowDataType,
) -> Box<dyn polars_arrow::array::Array> {
    let arrow_c_array = arrow::ffi::FFI_ArrowArray::new(&arrow_rs_array.to_data());
    let polars_c_array = unsafe { mem::transmute(arrow_c_array) };
    unsafe { polars_arrow::ffi::import_array_from_c(polars_c_array, polars_arrow_dtype) }.unwrap()
}

#[cfg(all(test, feature = "polars"))]
mod tests {
    use super::SendableRecordBatchStream;
    use crate::arrow::{
        IntoArrow, IntoPolars, PolarsDataFrameRecordBatchReader, SimpleRecordBatchStream,
    };
    use polars::df;

    fn get_record_batch_reader_from_polars() -> Box<dyn arrow_array::RecordBatchReader + Send> {
        let df1 = df!("string" => &["ab"],
             "int" => &[1],
             "float" => &[1.0])
        .unwrap();
        let df2 = df!("string" => &["bc"],
             "int" => &[2],
             "float" => &[2.0])
        .unwrap();

        PolarsDataFrameRecordBatchReader::new(df1.vstack(&df2).unwrap())
            .into_arrow()
            .unwrap()
    }

    #[test]
    fn from_polars_to_arrow() {
        let record_batch_reader = get_record_batch_reader_from_polars();
        let schema = record_batch_reader.schema();

        // Test schema conversion
        assert_eq!(
            schema
                .fields
                .iter()
                .map(|field| (field.name().as_str(), field.data_type()))
                .collect::<Vec<_>>(),
            vec![
                ("string", &arrow_schema::DataType::LargeUtf8),
                ("int", &arrow_schema::DataType::Int32),
                ("float", &arrow_schema::DataType::Float64)
            ]
        );
        let record_batches: Vec<arrow_array::RecordBatch> =
            record_batch_reader.map(|result| result.unwrap()).collect();
        assert_eq!(record_batches.len(), 2);
        assert_eq!(schema, record_batches[0].schema());
        assert_eq!(record_batches[0].schema(), record_batches[1].schema());

        // Test number of rows
        assert_eq!(record_batches[0].num_rows(), 1);
        assert_eq!(record_batches[1].num_rows(), 1);
    }

    #[tokio::test]
    async fn from_arrow_to_polars() {
        let record_batch_reader = get_record_batch_reader_from_polars();
        let schema = record_batch_reader.schema();
        let stream: SendableRecordBatchStream = Box::pin(SimpleRecordBatchStream {
            schema: schema.clone(),
            stream: futures::stream::iter(
                record_batch_reader
                    .into_iter()
                    .map(|r| r.map_err(Into::into)),
            ),
        });
        let df = stream.into_polars().await.unwrap();

        // Test number of chunks and rows
        assert_eq!(df.n_chunks(), 2);
        assert_eq!(df.height(), 2);

        // Test schema conversion
        assert_eq!(
            df.schema()
                .into_iter()
                .map(|(name, datatype)| (name.to_string(), datatype))
                .collect::<Vec<_>>(),
            vec![
                ("string".to_string(), polars::prelude::DataType::String),
                ("int".to_owned(), polars::prelude::DataType::Int32),
                ("float".to_owned(), polars::prelude::DataType::Float64)
            ]
        );
    }
}
